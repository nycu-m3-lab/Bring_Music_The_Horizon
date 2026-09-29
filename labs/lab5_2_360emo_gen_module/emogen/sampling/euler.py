import torch


def simple_euler_sampling(
    denoiser,
    guider,
    x,
    cond,
    uc,
    num_steps=25,
    sigma_min=0.0292,
    sigma_max=14.6146,
    rho=7.0,
    circular_padding_controller=None,
    circular_last_k_steps: int = 0,
):
    """Simple Euler sampling with SEGA guidance, with optional last-K circular padding."""
    device = x.device

    sigmas = torch.linspace(sigma_max ** (1 / rho), sigma_min ** (1 / rho), num_steps + 1, device=device) ** rho

    x = x * sigmas[0]
    guider.reset()

    if circular_padding_controller is not None:
        circular_padding_controller.disable()
    enable_from = max(0, num_steps - int(circular_last_k_steps or 0))

    for i in range(num_steps):
        if circular_padding_controller is not None and i == enable_from:
            circular_padding_controller.enable()

        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        x_in, sigma_in, cond_in = guider.prepare_inputs(x, sigma.repeat(x.shape[0]), cond, uc)
        denoised = denoiser(x_in, sigma_in, cond_in)
        denoised_guided = guider(x, denoised, sigma_in)

        d = (x - denoised_guided) / sigma
        dt = sigma_next - sigma
        x = x + d * dt

        if i % 5 == 0:
            print(f"Step {i+1}/{num_steps}, sigma={sigma:.4f}")

    if circular_padding_controller is not None:
        circular_padding_controller.disable()

    return x

