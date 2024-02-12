import torch
import torch.nn.functional as F

FIND_RED = torch.tensor([[1.0, 0.0, 0.0]])


def softmax_attention(Q, K, V, d_model):
    """
    Compute softmax attention given Q, K, V, and dimensionality d_model.
    """
    attention_scores = torch.matmul(
        Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
    attention_scores = F.softmax(attention_scores, dim=-1)
    return torch.matmul(attention_scores, V)


def main():
    num_images = 3

    R = torch.rand(num_images)  # Red channel values
    G = torch.rand(num_images)  # Green channel values
    B = torch.rand(num_images)  # Blue channel values

    pic = torch.stack([R, G, B], dim=1)  # Shape: [num_images, 3]
    print(f"Image: \n{pic.numpy()}")
    print(f"Image shape: {pic.numpy().shape}")

    # Assume K, V are the same as the input image
    K = V = pic  # Shape: [num_images, 3]

    # Assume Q is a single vector that we want to use to find red in the image
    Q = FIND_RED  # Shape: [1, 3]

    # Assume d_model is the dimensionality of the input image (3)
    d_model = 3

    # Compute attention
    attended_values = softmax_attention(Q, K, V, d_model)
    print(f"Attended values: {attended_values.numpy()}")


if __name__ == "__main__":
    main()
