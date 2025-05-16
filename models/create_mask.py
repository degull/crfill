import cv2
import numpy as np
import random
from PIL import Image, ImageDraw
import os
import math

class MaskCreator:
    def __init__(self, list_mask_path=None, base_mask_path=None, match_size=False):
        self.match_size = match_size
        if list_mask_path is not None and base_mask_path is not None:
            filenames = open(list_mask_path).readlines()
            self.msk_filenames = [os.path.join(base_mask_path, x.strip()) for x in filenames]
        else:
            self.msk_filenames = None

    def object_mask(self, image_height=256, image_width=256):
        # If no object mask files, fallback to random mask
        if self.msk_filenames is None or len(self.msk_filenames) == 0:
            return self.random_mask(image_height, image_width)

        hb, wb = image_height, image_width
        try:
            mask = Image.open(random.choice(self.msk_filenames)).convert("L")
        except Exception:
            return self.random_mask(image_height, image_width)

        wm, hm = mask.size
        r = float(min(hb, wb)) / max(wm, hm) / 2 if self.match_size else 1.0
        scale = min(2.0, max(0.5, random.gauss(r, 0.5)))
        wm, hm = int(wm * scale), int(hm * scale)
        mask = mask.resize((wm, hm))
        mask = (np.array(mask) > 0).astype(np.uint8)

        if mask.sum() == 0:
            return self.random_mask(image_height, image_width)

        col_nz = np.where(mask.sum(0) != 0)[0]
        row_nz = np.where(mask.sum(1) != 0)[0]
        mask = mask[row_nz[0]:row_nz[-1], col_nz[0]:col_nz[-1]]

        hm, wm = mask.shape
        canvas = np.zeros((hm + hb, wm + wb), dtype=np.uint8)
        y = random.randint(0, hb - 1)
        x = random.randint(0, wb - 1)
        canvas[y:y + hm, x:x + wm] = mask
        hole = canvas[hm // 2:hm // 2 + hb, wm // 2:wm // 2 + wb]

        th = 100 if self.match_size else 1000
        if hole.sum() < hb * wb / th:
            return self.random_mask(image_height, image_width)

        return hole.astype(np.float32)

    def random_mask(self, image_height=256, image_width=256, hole_range=[0, 1]):
        coef = min(hole_range[0] + hole_range[1], 1.0)
        while True:
            mask = np.ones((image_height, image_width), np.uint8)

            def Fill(max_size):
                w, h = np.random.randint(max_size), np.random.randint(max_size)
                ww, hh = w // 2, h // 2
                x, y = np.random.randint(-ww, image_width - w + ww), np.random.randint(-hh, image_height - h + hh)
                mask[max(y, 0): min(y + h, image_height), max(x, 0): min(x + w, image_width)] = 0

            def MultiFill(max_tries, max_size):
                for _ in range(np.random.randint(max_tries)):
                    Fill(max_size)

            MultiFill(int(10 * coef), max(image_height, image_width) // 2)
            MultiFill(int(5 * coef), max(image_height, image_width))
            mask = np.logical_and(mask, 1 - self.random_brush(int(20 * coef), image_height, image_width))
            hole_ratio = 1 - np.mean(mask)
            if hole_ratio >= hole_range[0] and hole_ratio <= hole_range[1]:
                break
        return 1 - mask.astype(np.float32)

    def random_brush(self, max_tries, image_height=256, image_width=256, min_num_vertex=4,
                     max_num_vertex=18, mean_angle=2 * math.pi / 5, angle_range=2 * math.pi / 15,
                     min_width=12, max_width=48):
        H, W = image_height, image_width
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(max_tries)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []

            for i in range(num_vertex):
                angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max) if i % 2 == 0 else np.random.uniform(angle_min, angle_max))

            vertex.append((np.random.randint(0, W), np.random.randint(0, H)))
            for angle in angles:
                r = np.clip(np.random.normal(loc=average_radius, scale=average_radius // 2), 0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angle), 0, W)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angle), 0, H)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = np.random.randint(min_width, max_width)
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=1)

        if random.random() > 0.5:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return np.asarray(mask, np.uint8)
    
    def stroke_mask(self, image_height=256, image_width=256, max_vertex=5, max_mask=5, max_length=128):
        max_angle = np.pi
        max_brush_width = max(1, int(max_length*0.4))
        min_brush_width = max(1, int(max_length*0.1))

        mask = np.zeros((image_height, image_width))
        for _ in range(random.randint(1, max_mask)):
            num_vertex = random.randint(1, max_vertex)
            start_x = random.randint(0, image_width-1)
            start_y = random.randint(0, image_height-1)
            for _ in range(num_vertex):
                angle = random.uniform(0, max_angle)
                angle = 2*np.pi - angle if random.randint(0,1) else angle
                length = random.uniform(0, max_length)
                brush_width = random.randint(min_brush_width, max_brush_width)
                end_x = min(int(start_x + length * np.cos(angle)), image_width-1)
                end_y = min(int(start_y + length * np.sin(angle)), image_height-1)
                mask = cv2.line(mask, (start_x, start_y), (end_x, end_y), color=1, thickness=brush_width)
                start_x, start_y = end_x, end_y
                mask = cv2.circle(mask, (start_x, start_y), int(brush_width/2), 1)
            if random.randint(0, 1):
                mask = mask[:, ::-1].copy()
            if random.randint(0, 1):
                mask = mask[::-1, :].copy()
        return mask

    def rectangle_mask(self, image_height=256, image_width=256, min_hole_size=64, max_hole_size=128):
        mask = np.zeros((image_height, image_width))
        hole_size = random.randint(min_hole_size, max_hole_size)
        hole_size = min(int(image_width*0.8), int(image_height*0.8), hole_size)
        x = random.randint(0, image_width - hole_size - 1)
        y = random.randint(0, image_height - hole_size - 1)
        mask[y:y+hole_size, x:x+hole_size] = 1
        return mask



if __name__ == "__main__":
    mask_creator = MaskCreator()
    mask = mask_creator.random_mask(image_height=512, image_width=512)
    Image.fromarray((mask * 255).astype(np.uint8)).save("output/mask.png")
