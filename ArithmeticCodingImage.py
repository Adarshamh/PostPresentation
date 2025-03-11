import os
import cv2
import numpy as np
import collections
import matplotlib.pyplot as plt
from PIL import Image
import pickle

resolution= (1080, 720)
# Arithmetic Coding Compression
class ArithmeticCoding:
    def __init__(self, data):
        self.frequencies = collections.Counter(data)
        self.total_symbols = sum(self.frequencies.values())
        self.intervals = self.calculate_intervals()
    
    def calculate_intervals(self):
        low = 0.0
        intervals = {}
        for symbol, freq in sorted(self.frequencies.items()):
            high = low + (freq / self.total_symbols)
            intervals[symbol] = (low, high)
            low = high
        return intervals
    
    def encode(self, data):
        low, high = 0.0, 1.0
        for symbol in data:
            symbol_low, symbol_high = self.intervals[symbol]
            range_ = high - low
            high = low + range_ * symbol_high
            low = low + range_ * symbol_low
        return (low + high) / 2  # Encoded value
    
    def decode(self, encoded_value, data_length):
        decoded_data = []
        for _ in range(data_length):
            for symbol, (symbol_low, symbol_high) in self.intervals.items():
                if symbol_low <= encoded_value < symbol_high:
                    decoded_data.append(symbol)
                    range_ = symbol_high - symbol_low
                    encoded_value = (encoded_value - symbol_low) / range_
                    break
        return decoded_data

# Compress Image Using Arithmetic Coding
def arithmetic_compress(image_path, output_file):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_shape = image.shape  # Store original resolution

    flattened = image.flatten()
    encoder = ArithmeticCoding(flattened)
    encoded_value = encoder.encode(flattened)
    
    with open(output_file, "wb") as f:
        pickle.dump((encoded_value, encoder.frequencies, original_shape), f)  # Save original shape
    
    return os.path.getsize(output_file) / 1024, original_shape  # Return file size and resolution

# Decompress Arithmetic-encoded image
def arithmetic_decompress(input_file, output_path):
    with open(input_file, "rb") as f:
        encoded_value, frequencies, original_shape = pickle.load(f)
    
    decoder = ArithmeticCoding([])
    decoder.frequencies = frequencies
    decoder.total_symbols = sum(frequencies.values())
    decoder.intervals = decoder.calculate_intervals()
    
    decoded_data = decoder.decode(encoded_value, np.prod(original_shape))
    decompressed_image = np.array(decoded_data, dtype=np.uint8).reshape(original_shape)  # Restore original resolution
    cv2.imwrite(output_path, decompressed_image)

# Load Image
input_image = "C:\\Users\\Adars\\Desktop\\image\\Input\\vividly-colored-hummingbird-nature.jpg"
arithmetic_output = "arithmetic_compressed.bin"
decompressed_output = "arithmetic_decompressed.jpg"

# Apply Arithmetic Compression
arithmetic_size, original_shape = arithmetic_compress(input_image, arithmetic_output)

# Decompress Image
arithmetic_decompress(arithmetic_output, decompressed_output)

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
print(f"Arithmetic Decompressed Size: {decompressed_size:.2f} KB")
print(f"Arithmetic Compressed Size: {arithmetic_size:.2f} KB")

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
axes[4].set_title(f"Arithmetic Decompressed ({decompressed_size:.2f} KB, {resolution})")
axes[4].axis("off")

plt.savefig("arithmetic_output.png")
