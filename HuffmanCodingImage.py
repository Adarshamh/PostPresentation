import os
import cv2
import numpy as np
import heapq
import collections
import matplotlib.pyplot as plt
from PIL import Image
import pickle

resolution = (1080, 720)

# Huffman Node class
class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ""
    
    def __lt__(self, other):
        return self.freq < other.freq

# Build Huffman Tree
def build_huffman_tree(data):
    frequency = collections.Counter(data)
    heap = [Node(freq, symbol) for symbol, freq in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
        heapq.heappush(heap, merged)
    
    return heap[0]

# Generate Huffman Codes
def generate_codes(node, current_code="", code_map={}):
    if node:
        if not node.left and not node.right:
            code_map[node.symbol] = current_code
        generate_codes(node.left, current_code + "0", code_map)
        generate_codes(node.right, current_code + "1", code_map)
    return code_map

# Compress Image Using Huffman Coding
def huffman_compress(image_path, output_file):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_shape = image.shape  # Store original resolution

    flattened = image.flatten()
    root = build_huffman_tree(flattened)
    code_map = generate_codes(root)

    encoded_data = "".join(code_map[pixel] for pixel in flattened)
    padding = 8 - len(encoded_data) % 8
    encoded_data += "0" * padding

    byte_array = bytearray()
    for i in range(0, len(encoded_data), 8):
        byte_array.append(int(encoded_data[i:i+8], 2))

    with open(output_file, "wb") as f:
        pickle.dump((byte_array, code_map, padding, original_shape), f)  # Save original shape

    return os.path.getsize(output_file) / 1024, original_shape  # Return file size and resolution

# Decompress Huffman-encoded image
def huffman_decompress(input_file, output_path):
    with open(input_file, "rb") as f:
        byte_array, code_map, padding, original_shape = pickle.load(f)

    binary_data = "".join(f"{byte:08b}" for byte in byte_array)
    binary_data = binary_data[:-padding]  # Remove padding

    reverse_map = {v: k for k, v in code_map.items()}
    decoded_data = []
    temp_code = ""

    for bit in binary_data:
        temp_code += bit
        if temp_code in reverse_map:
            decoded_data.append(reverse_map[temp_code])
            temp_code = ""

    decompressed_image = np.array(decoded_data, dtype=np.uint8).reshape(original_shape)  # Restore original resolution
    cv2.imwrite(output_path, decompressed_image)

# Load Image
input_image = "C:\\Users\\Adars\\Desktop\\image\\Input\\vividly-colored-hummingbird-nature.jpg"
huffman_output = "huffman_compressed.bin"
decompressed_output = "huffman_decompressed.jpg"

# Apply Huffman Compression
huffman_size, original_shape = huffman_compress(input_image, huffman_output)

# Decompress Image
huffman_decompress(huffman_output, decompressed_output)

# Save Compressed Images in Different Formats
image = Image.open(input_image)

jpeg_output = "lossy_compressed.jpg"
image.save(jpeg_output, "JPEG", quality=10)

png_output = "lossless_compressed.png"
image.save(png_output, "PNG", optimize=True)

webp_output = "lossless_compressed.webp"
image.save(webp_output, "WEBP", lossless=True)

# Get File Sizes
original_size = os.path.getsize(input_image) / 1024  # KB
jpeg_size = os.path.getsize(jpeg_output) / 1024  # KB
png_size = os.path.getsize(png_output) / 1024  # KB
webp_size = os.path.getsize(webp_output) / 1024  # KB
decompressed_size = os.path.getsize(decompressed_output) / 1024  # KB

# Print File Sizes
print(f"Original Size: {original_size:.2f} KB")
print(f"JPEG Compressed Size: {jpeg_size:.2f} KB")
print(f"PNG Compressed Size: {png_size:.2f} KB")
print(f"WebP Compressed Size: {webp_size:.2f} KB")
print(f"Huffman Decompressed Size: {decompressed_size:.2f} KB")
print(f"Huffman Compressed Size: {huffman_size:.2f} KB")

# Load Images for Display
original_img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)  # Keep original resolution
decompressed_img = cv2.imread(decompressed_output, cv2.IMREAD_GRAYSCALE)
jpeg_img = cv2.imread(jpeg_output, cv2.IMREAD_GRAYSCALE)
png_img = cv2.imread(png_output, cv2.IMREAD_GRAYSCALE)
webp_img = cv2.imread(webp_output, cv2.IMREAD_GRAYSCALE)

# Resize compressed images dynamically based on original image resolution
def resize_image(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))  # Resize keeping aspect ratio

decompressed_img = resize_image(decompressed_img, resolution)
jpeg_img = resize_image(jpeg_img, resolution)
png_img = resize_image(png_img, resolution)
webp_img = resize_image(webp_img, resolution)

# Display Images
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

axes[0].imshow(original_img, cmap='gray')
axes[0].set_title(f"Original (JPEG, {original_size:.2f} KB, {original_shape})")
axes[0].axis("off")

axes[1].imshow(jpeg_img, cmap='gray')
axes[1].set_title(f"Lossy JPEG ({jpeg_size:.2f} KB, {resolution})")
axes[1].axis("off")

axes[2].imshow(png_img, cmap='gray')
axes[2].set_title(f"Lossless PNG ({png_size:.2f} KB, {resolution})")
axes[2].axis("off")

axes[3].imshow(webp_img, cmap='gray')
axes[3].set_title(f"Lossless WebP ({webp_size:.2f} KB, {resolution})")
axes[3].axis("off")

axes[4].imshow(decompressed_img, cmap='gray')
axes[4].set_title(f"Huffman Decompressed ({decompressed_size:.2f} KB, {resolution})")
axes[4].axis("off")

plt.savefig("huffman_output.png")
