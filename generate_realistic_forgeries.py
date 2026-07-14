import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def create_base_document():
    size = 512
    base_color = random.randint(240, 252)
    img_array = np.full((size, size, 3), base_color, dtype=np.float32)

    for i in range(size):
        gradient = (i / size) * random.uniform(-6, 6)
        img_array[i, :, :] += gradient

    noise = np.random.normal(0, random.uniform(1.5, 3.5), (size, size, 3))
    img_array += noise
    img_array = np.clip(img_array, 220, 255).astype(np.uint8)
    image = Image.fromarray(img_array, 'RGB')
    draw = ImageDraw.Draw(image)

    doc_type = random.choice(['receipt', 'invoice', 'letter'])

    if doc_type == 'receipt':
        y = random.randint(20, 40)
        header_w = random.randint(180, 280)
        header_h = random.randint(10, 16)
        x_start = (size - header_w) // 2
        draw.rectangle([x_start, y, x_start + header_w, y + header_h],
                       fill=random.randint(30, 60))
        y += header_h + random.randint(6, 12)

        sub_w = random.randint(100, 160)
        sub_h = random.randint(5, 8)
        draw.rectangle([(size - sub_w) // 2, y, (size + sub_w) // 2, y + sub_h],
                       fill=random.randint(80, 120))
        y += sub_h + random.randint(10, 18)

        draw.line([(30, y), (size - 30, y)], fill=random.randint(150, 180), width=1)
        y += random.randint(8, 14)

        for _ in range(random.randint(5, 9)):
            left_w = random.randint(120, 200)
            right_w = random.randint(40, 70)
            h = random.randint(4, 7)
            noise_offset = random.randint(-1, 1)
            draw.rectangle([30, y + noise_offset, 30 + left_w, y + h + noise_offset],
                           fill=random.randint(40, 80))
            draw.rectangle([size - 30 - right_w, y, size - 30, y + h],
                           fill=random.randint(40, 80))
            y += h + random.randint(7, 13)

        draw.line([(30, y), (size - 30, y)], fill=random.randint(150, 180), width=1)
        y += random.randint(6, 10)

        total_w = random.randint(80, 120)
        total_h = random.randint(7, 11)
        draw.rectangle([size - 30 - total_w, y, size - 30, y + total_h],
                       fill=random.randint(20, 50))
        y += total_h + random.randint(12, 20)

        for _ in range(random.randint(2, 4)):
            w = random.randint(60, 140)
            h = random.randint(3, 5)
            x = random.randint(30, size - 30 - w)
            draw.rectangle([x, y, x + w, y + h], fill=random.randint(60, 100))
            y += h + random.randint(5, 9)

    elif doc_type == 'invoice':
        draw.rectangle([25, 20, size - 25, 55], fill=random.randint(30, 70))

        for col_x in [30, 150, 280, 380]:
            w = random.randint(60, 100)
            h = random.randint(5, 9)
            draw.rectangle([col_x, 65, col_x + w, 65 + h], fill=random.randint(40, 90))

        draw.line([(25, 80), (size - 25, 80)], fill=random.randint(120, 160), width=2)

        y = 90
        for _ in range(random.randint(6, 10)):
            for col_x in [30, 150, 280, 380]:
                w = random.randint(50, 90)
                h = random.randint(4, 6)
                noise_x = random.randint(-2, 2)
                draw.rectangle([col_x + noise_x, y, col_x + w + noise_x, y + h],
                               fill=random.randint(50, 100))
            draw.line([(25, y + 10), (size - 25, y + 10)],
                      fill=random.randint(200, 220), width=1)
            y += random.randint(14, 20)

        draw.line([(25, y + 5), (size - 25, y + 5)],
                  fill=random.randint(100, 140), width=2)
        y += 12
        draw.rectangle([size - 160, y, size - 25, y + 14],
                       fill=random.randint(20, 60))

        y += 30
        for _ in range(random.randint(3, 5)):
            w = random.randint(80, 180)
            h = random.randint(3, 5)
            draw.rectangle([30, y, 30 + w, y + h], fill=random.randint(60, 110))
            y += h + random.randint(6, 10)

    else:
        y = 25
        for _ in range(random.randint(2, 3)):
            w = random.randint(100, 200)
            h = random.randint(6, 10)
            draw.rectangle([30, y, 30 + w, y + h], fill=random.randint(30, 70))
            y += h + random.randint(4, 8)

        y += random.randint(10, 18)
        draw.line([(30, y), (size - 30, y)], fill=random.randint(160, 200), width=1)
        y += random.randint(10, 16)

        for _ in range(random.randint(4, 6)):
            num_words = random.randint(4, 8)
            x = 30
            for _ in range(num_words):
                w = random.randint(20, 55)
                h = random.randint(4, 6)
                noise_y = random.randint(-1, 1)
                draw.rectangle([x, y + noise_y, x + w, y + h + noise_y],
                               fill=random.randint(40, 80))
                x += w + random.randint(4, 10)
                if x > size - 60:
                    break
            y += random.randint(10, 16)

        y += random.randint(8, 14)
        sig_w = random.randint(80, 140)
        sig_h = random.randint(6, 10)
        draw.rectangle([30, y, 30 + sig_w, y + sig_h],
                       fill=random.randint(20, 60))

    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.5)))
    return image


def apply_tampering(image):
    size = 512
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    img_array = np.array(image.copy())
    technique = random.choice(['copy_move', 'splicing'])

    if technique == 'copy_move':
        src_w = random.randint(60, 160)
        src_h = random.randint(30, 100)
        src_x = random.randint(20, size - src_w - 20)
        src_y = random.randint(20, size - src_h - 20)

        patch = img_array[src_y:src_y + src_h, src_x:src_x + src_w].copy()

        max_offset_x = size - src_w - 20
        max_offset_y = size - src_h - 20
        dst_x = random.randint(20, max(20, max_offset_x))
        dst_y = random.randint(20, max(20, max_offset_y))

        while (abs(dst_x - src_x) < 20 and abs(dst_y - src_y) < 20):
            dst_x = random.randint(20, max(20, max_offset_x))
            dst_y = random.randint(20, max(20, max_offset_y))

        brightness_shift = np.random.randint(-12, 12, patch.shape).astype(np.int16)
        pasted = np.clip(patch.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)

        actual_h = min(src_h, size - dst_y)
        actual_w = min(src_w, size - dst_x)
        img_array[dst_y:dst_y + actual_h, dst_x:dst_x + actual_w] = pasted[:actual_h, :actual_w]

        mask_draw.rectangle([dst_x, dst_y, dst_x + actual_w, dst_y + actual_h], fill=255)

    else:
        x1 = random.randint(20, 280)
        y1 = random.randint(20, 380)
        region_w = random.randint(80, 200)
        region_h = random.randint(40, 120)
        x2 = min(x1 + region_w, size - 20)
        y2 = min(y1 + region_h, size - 20)

        bg_sample = img_array[y1:y2, x1:x2].mean()
        splice_bg = int(np.clip(bg_sample + random.choice([-1, 1]) * random.uniform(8, 22), 200, 252))
        img_array[y1:y2, x1:x2] = splice_bg

        splice_img = Image.fromarray(img_array)
        splice_draw = ImageDraw.Draw(splice_img)
        ink_color = random.randint(30, 85)

        y_cursor = y1 + random.randint(5, 10)
        while y_cursor < y2 - 8:
            num_blocks = random.randint(3, 7)
            x_cursor = x1 + random.randint(4, 10)
            for _ in range(num_blocks):
                bw = random.randint(15, 45)
                bh = random.randint(4, 7)
                if x_cursor + bw > x2 - 4:
                    break
                noise_y = random.randint(-1, 1)
                splice_draw.rectangle(
                    [x_cursor, y_cursor + noise_y, x_cursor + bw, y_cursor + bh + noise_y],
                    fill=ink_color
                )
                x_cursor += bw + random.randint(4, 9)
            y_cursor += random.randint(9, 14)

        img_array = np.array(splice_img)

        region = Image.fromarray(img_array[y1:y2, x1:x2])
        region = region.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.9)))
        img_array[y1:y2, x1:x2] = np.array(region)

        mask_draw.rectangle([x1, y1, x2, y2], fill=255)

    tampered_image = Image.fromarray(img_array)
    return tampered_image, mask


def apply_compression_artifacts(image):
    blur_radius = random.uniform(0.5, 1.0)
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return image


if __name__ == '__main__':
    out_img_dir = 'data/synthetic/images'
    out_msk_dir = 'data/synthetic/masks'
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    total = 100
    forged_count = 0
    authentic_count = 0

    for i in range(total):
        base = create_base_document()
        mask = Image.new('L', (512, 512), 0)

        if random.random() < 0.8:
            rgb, mask = apply_tampering(base)
            forged_count += 1
        else:
            rgb = base
            authentic_count += 1

        rgb = apply_compression_artifacts(rgb)

        quality = random.randint(65, 90)
        img_path = os.path.join(out_img_dir, f'synthetic_{i:04d}.jpg')
        msk_path = os.path.join(out_msk_dir, f'synthetic_{i:04d}.png')
        rgb.save(img_path, 'JPEG', quality=quality)
        mask.save(msk_path, 'PNG')

        filled = int((i + 1) / total * 40)
        bar = '#' * filled + '-' * (40 - filled)
        print(f'\r[{bar}] {i + 1}/{total}  forged={forged_count}  authentic={authentic_count}', end='', flush=True)

    print(f'\nDone. {forged_count} forged, {authentic_count} authentic.')
    print(f'Images → {out_img_dir}')
    print(f'Masks  → {out_msk_dir}')
