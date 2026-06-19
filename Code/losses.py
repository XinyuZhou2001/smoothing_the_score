import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  elif config.optim.optimizer == 'AdamW':
      optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                             weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def setup_feature_extractor(device='cuda', feature_type='resnet18'):
    """Setup feature extractor for k-NN computation.
    
    Args:
        device: Computation device
        feature_type: Type of feature extractor ('resnet18', 'resnet34', 'resnet50', 'vgg16', 'efficientnet')
        
    Returns:
        feature_extractor_fn: Feature extraction function
        feature_dim: Feature dimension
    """
    
    if feature_type == 'resnet18':
        model = models.resnet18(pretrained=True)
        feature_dim = 512
        # Remove final fully connected layer and average pooling layer
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    elif feature_type == 'resnet34':
        model = models.resnet34(pretrained=True)
        feature_dim = 512
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    elif feature_type == 'resnet50':
        model = models.resnet50(pretrained=True)
        feature_dim = 2048
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    elif feature_type == 'vgg16':
        model = models.vgg16(pretrained=True)
        feature_dim = 512
        # Use VGG features part
        feature_extractor = model.features
    elif feature_type == 'efficientnet':
        try:
            model = models.efficientnet_b0(pretrained=True)
            feature_dim = 1280
            feature_extractor = model.features
        except AttributeError:
            # Fallback to resnet18 if efficientnet not available
            print("EfficientNet not available, falling back to ResNet18")
            model = models.resnet18(pretrained=True)
            feature_dim = 512
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Freeze parameters
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    def extract_features(images):
        """Feature extraction function"""
        with torch.no_grad():
            # Ensure input is in correct range [0,1]
            if images.min() < 0:
                images = (images + 1) / 2.0
            
            # Data preprocessing: ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            
            # For CIFAR-10 (32x32), resize to appropriate size
            if images.shape[-1] == 32:
                # For 32x32 images, use bilinear interpolation to 224x224
                images = F.interpolate(images, size=(224, 224), 
                                     mode='bilinear', align_corners=False)
            elif images.shape[-1] != 224:
                # Resize other sizes to 224x224
                images = F.interpolate(images, size=(224, 224), 
                                     mode='bilinear', align_corners=False)
            
            # Normalization
            images = (images - mean) / std
            
            # Extract features
            features = feature_extractor(images)
            
            # Global average pooling
            if features.dim() == 4:  # [B, C, H, W]
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            
            # L2 normalization
            features = F.normalize(features, p=2, dim=1)
            
        return features
    
    print(f"Using {feature_type} feature extractor, feature dimension: {feature_dim}")
    return extract_features, feature_dim


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbitrary SDEs."""
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False, all_train_data=None, 
                     device='cuda', feature_type='resnet18'):
  """Modified SMLD loss with k-nearest neighbor score matching in feature space."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  # Ensure training data is provided
  if all_train_data is None:
    raise ValueError("all_train_data is required for k-NN score matching")

  # Setup feature extractor
  feature_extractor, feature_dim = setup_feature_extractor(device, feature_type)
  


  # k-NN related parameters
  k = 1
  temperature_factor = 0



  feature_space_verified = False  # Flag for one-time verification





  # Precompute features for all training data
  print("Precomputing training data features...")
  with torch.no_grad():
    all_train_features = []
    batch_size = 256  # Batch processing to avoid memory overflow
    
    for i in range(0, all_train_data.shape[0], batch_size):
      batch_end = min(i + batch_size, all_train_data.shape[0])
      batch_data = all_train_data[i:batch_end]
      batch_features = feature_extractor(batch_data)
      all_train_features.append(batch_features)
    
    all_train_features = torch.cat(all_train_features, dim=0)
    print(f"Training data feature precomputation completed: {all_train_features.shape}")

  def compute_knn_indices_feature_chunked(samples, training_features, k=50, chunk_size=5000):
    """Compute k-nearest neighbor indices in feature space with chunking."""
    nonlocal feature_space_verified
    
    batch_size = samples.shape[0]
    total_samples = training_features.shape[0]
    
    # Compute features for query samples
    with torch.no_grad():
      query_features = feature_extractor(samples)
      
      # One-time verification
      if not feature_space_verified:
        pixel_dims = np.prod(samples.shape[1:])  # C*H*W
        feature_dims = query_features.shape[1]   # Feature dimensions
        print(f"VERIFICATION: k-NN search in {feature_dims}D feature space (vs {pixel_dims}D pixel space)")
        print(f"VERIFICATION: Feature dimension reduction ratio: {pixel_dims/feature_dims:.1f}x")
        feature_space_verified = True
    
    # Initialize result tensors
    final_distances = torch.full((batch_size, k), float('inf'), device=samples.device)
    final_indices = torch.zeros((batch_size, k), dtype=torch.long, device=samples.device)
    
    with torch.no_grad():
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_features = training_features[start_idx:end_idx]
            
            # Compute cosine similarity distance (features are L2 normalized)
            cosine_sim = torch.mm(query_features, chunk_features.t())
            chunk_distances = 1.0 - cosine_sim
            
            chunk_k = min(k, chunk_distances.shape[1])
            chunk_topk_dist, chunk_topk_idx = torch.topk(
                chunk_distances, chunk_k, dim=1, largest=False
            )
            
            chunk_topk_idx += start_idx
            
            combined_distances = torch.cat([final_distances, chunk_topk_dist], dim=1)
            combined_indices = torch.cat([final_indices, chunk_topk_idx], dim=1)
            
            _, top_k_positions = torch.topk(combined_distances, k, dim=1, largest=False)
            
            batch_indices = torch.arange(batch_size)[:, None].expand(-1, k).to(samples.device)
            final_distances = combined_distances[batch_indices, top_k_positions]
            final_indices = combined_indices[batch_indices, top_k_positions]
    
    return final_indices

  def compute_knn_indices_pixel_chunked(samples, training_data, k=50, chunk_size=10000):
    """k-nearest neighbor computation in pixel space (fallback)."""
    batch_size = samples.shape[0]
    total_samples = training_data.shape[0]
    
    samples_flat = samples.reshape(batch_size, -1)
    training_flat = training_data.reshape(total_samples, -1)
    
    print(f"WARNING: Fallback to pixel space k-NN computation ({samples_flat.shape[1]}D)")
    
    final_distances = torch.full((batch_size, k), float('inf'), device=samples.device)
    final_indices = torch.zeros((batch_size, k), dtype=torch.long, device=samples.device)
    
    with torch.no_grad():
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_data = training_flat[start_idx:end_idx]
            
            chunk_distances = torch.cdist(samples_flat, chunk_data, p=2)
            
            chunk_k = min(k, chunk_distances.shape[1])
            chunk_topk_dist, chunk_topk_idx = torch.topk(
                chunk_distances, chunk_k, dim=1, largest=False
            )
            
            chunk_topk_idx += start_idx
            
            combined_distances = torch.cat([final_distances, chunk_topk_dist], dim=1)
            combined_indices = torch.cat([final_indices, chunk_topk_idx], dim=1)
            
            _, top_k_positions = torch.topk(combined_distances, k, dim=1, largest=False)
            
            batch_indices = torch.arange(batch_size)[:, None].expand(-1, k).to(samples.device)
            final_distances = combined_distances[batch_indices, top_k_positions]
            final_indices = combined_indices[batch_indices, top_k_positions]
    
    return final_indices

  def compute_explicit_score(perturbed_samples, knn_centers, used_sigmas, temperature_factor=10.0):
    """Compute explicit score function."""
    d = np.prod(perturbed_samples.shape[1:])  # Total dimension count
    batch_size, k = knn_centers.shape[0], knn_centers.shape[1]

    x_expanded = perturbed_samples.unsqueeze(1)
    diffs = x_expanded - knn_centers
    distances_sq = (diffs.reshape(batch_size, k, -1) ** 2).sum(dim=-1)

    sigma_flat = used_sigmas.reshape(batch_size, 1)
    sigma_sq = sigma_flat ** 2
    #temperature = (temperature_factor)/(sigma_flat)
    temperature = d**0.5

    log_weights = -d / (2 * temperature) * torch.log(distances_sq / d + 1e-10)

    max_log_weights = log_weights.max(dim=1, keepdim=True)[0]
    normalized_log_weights = log_weights - max_log_weights
    weights = torch.exp(normalized_log_weights)
    weights_normalized = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)

    component_scores = -diffs / sigma_sq.reshape(batch_size, 1, *([1] * len(diffs.shape[2:])))

    weights_expanded = weights_normalized.reshape(batch_size, k, *([1] * len(component_scores.shape[2:])))
    mixture_score = (weights_expanded * component_scores).sum(dim=1)

    return mixture_score

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch

    score = model_fn(perturbed_data, labels)

    sigma_values = sigmas
    low_noise_mask = (sigma_values < temperature_factor)
    high_noise_mask = ~low_noise_mask

    total_losses = torch.zeros(batch.shape[0], dtype=batch.dtype, device=batch.device)

    # Handle high noise samples
    if high_noise_mask.any():
      high_score = score[high_noise_mask]
      high_noise = noise[high_noise_mask]
      high_sigmas = sigmas[high_noise_mask]

      target = -high_noise / (high_sigmas ** 2)[:, None, None, None]
      high_losses = torch.square(high_score - target)
      high_losses = reduce_op(high_losses.reshape(high_losses.shape[0], -1), dim=-1) * (high_sigmas ** 2)

      total_losses[high_noise_mask] = high_losses.to(batch.dtype)

    # Handle low noise samples
    if low_noise_mask.any():
      low_batch = batch[low_noise_mask]
      low_perturbed = perturbed_data[low_noise_mask]
      low_score = score[low_noise_mask]
      low_sigmas = sigmas[low_noise_mask]

      # Compute k-nearest neighbors in feature space
      try:
        knn_indices = compute_knn_indices_feature_chunked(
            low_batch, all_train_features, k=k
        )
      except Exception as e:
        print(f"Feature space k-NN computation failed, falling back to pixel space: {e}")
        knn_indices = compute_knn_indices_pixel_chunked(
            low_batch, all_train_data, k=k
        )

      # Get k-NN centers in pixel space for score computation
      knn_centers = all_train_data[knn_indices]

      true_score = compute_explicit_score(
        low_perturbed,
        knn_centers,
        low_sigmas[:, None, None, None],
        temperature_factor
      )

      low_losses = torch.square(low_score - true_score)
      low_losses = reduce_op(low_losses.reshape(low_losses.shape[0], -1), dim=-1) * (low_sigmas ** 2)

      total_losses[low_noise_mask] = low_losses.to(batch.dtype)

    loss = torch.mean(total_losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, 
                likelihood_weighting=False, all_train_data=None, device='cuda', 
                feature_type='resnet18'):
  """Get step function with feature space support."""
  
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean, 
                               all_train_data=all_train_data, device=device, 
                               feature_type=feature_type)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn