import torch
import torch.nn.functional as F

NUM_IMAGES = 1_000


def softmax_attention(Q, K, V, d_model):
    """
    Apply softmax attention and return attended values and attention weights.
    """
    attention_scores = torch.matmul(
        Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
    attention_weights = F.softmax(attention_scores, dim=-1)
    return torch.matmul(attention_weights, V), attention_weights


def find_max_yellow_pic(pics):
    """
    Find the index of the image with the maximum yellow content (red + green).
    """
    yellow_values = pics[:, 0] + pics[:, 1]
    max_yellow_index = torch.argmax(yellow_values)
    return max_yellow_index


def find_max_red_pic(pics):
    """
    Find the index of the image with the maximum red content.
    """
    max_red_index = torch.argmax(pics[:, 0])
    return max_red_index


def simulate_attention_vs_direct_comparison(query=torch.tensor([1.0, 0.0, 0.0]), TYPE="red"):
    # Simulate RGB values for a large number of images.
    R, G, B = torch.rand(NUM_IMAGES, 3).unbind(-1)
    pics = torch.stack([R, G, B], dim=1)

    # Define query
    Q = query

    # Compute attention
    attended_values, attention_weights = softmax_attention(
        Q, pics, pics, d_model=3)

    # Determine the most correlating image according to both methods.
    direct_max_index = find_max_yellow_pic(
        pics) if TYPE == "yellow" else find_max_red_pic(pics)
    attention_max_index = torch.argmax(attention_weights)
    return direct_max_index == attention_max_index


def main():
    query = torch.tensor([1.0, 0.0, 0.0])
    TYPE = "red"
    if TYPE == "yellow":
        query = torch.tensor([1.0, 1.0, 0.0])
    epochs = 10
    correct_matches = sum(simulate_attention_vs_direct_comparison(query, TYPE)
                          for _ in range(epochs))

    accuracy = correct_matches / epochs * 100
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
