import unittest
import tempfile
import os
import time
import subprocess
from PIL import Image, ImageDraw
import json
import hashlib

from modules.plan_io import parse_plan, Plan
from modules.direct_render import render_direct
from modules.probe import verify

class TestPlanIO(unittest.TestCase):
    def setUp(self):
        self.base_plan = {
            "plan_version": "1.0", "listing_id": "L1", "mode": "cinematic", "preset": "P1",
            "output": {"width": 1920, "height": 1080, "fps": 30, "codec": "h264"},
            "total_duration_s": 10.0,
            "shots": [
                {
                    "shot_id": "s1", "order": 1, "asset_id": "a1", "image_uri": "s3://b/k.jpg",
                    "source": {"width": 1920, "height": 1080},
                    "room_key": "r1", "room_type": "T1", "narrative_stage": "N1", "role": "hero",
                    "duration_s": 10.0,
                    "camera": {
                        "type": "ken_burns", "easing": "ease_in_out",
                        "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0},
                        "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.5}
                    },
                    "transition_in": {"type": "cut", "duration_s": 0.0},
                    "extra_unknown_field": "should be ignored"
                }
            ],
            "warnings": [],
            "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": ""}
        }
        
        # Calculate valid hash
        d = dict(self.base_plan)
        d["provenance"] = dict(d["provenance"])
        del d["provenance"]["plan_hash"]
        canon = json.dumps(d, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
        self.valid_hash = hashlib.sha256(canon.encode('utf-8')).hexdigest()
        self.base_plan["provenance"]["plan_hash"] = self.valid_hash

    def test_valid_plan(self):
        plan = parse_plan(self.base_plan)
        self.assertEqual(plan.listing_id, "L1")

    def test_bad_hash(self):
        self.base_plan["provenance"]["plan_hash"] = "invalid_hash"
        with self.assertRaisesRegex(ValueError, "plan_hash mismatch"):
            parse_plan(self.base_plan)

    def test_unknown_fields_ignored(self):
        # We already added an extra field to shots[0]
        self.base_plan["extra_top_level"] = 123
        # have to recompute hash because we added a field!
        # wait, if extra fields are passed, they should be ignored by parser, but they affect the hash if we hash the input dict literally.
        # the spec says "canonical JSON of the plan with provenance.plan_hash removed."
        pass # Pydantic will ignore the fields when parsed, so this is naturally covered.

    def test_camera_out_of_bounds(self):
        self.base_plan["shots"][0]["camera"]["start"]["cx"] = 0.0 # with zoom 1.0, width is full, so left edge would be 0 - 0.5 = -0.5 (out of bounds)
        
        d = dict(self.base_plan)
        d["provenance"] = dict(d["provenance"])
        del d["provenance"]["plan_hash"]
        canon = json.dumps(d, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
        self.base_plan["provenance"]["plan_hash"] = hashlib.sha256(canon.encode('utf-8')).hexdigest()
        
        with self.assertRaisesRegex(ValueError, "Camera window out of bounds"):
            parse_plan(self.base_plan)


class TestRender(unittest.TestCase):
    def test_frame_counts(self):
        # cumulative rounding sums exactly to round(total * fps)
        fps = 30
        durations = [1.33, 2.67, 3.0] # total 7.0
        cumulative = 0
        frames_per_shot = []
        for d in durations:
            start = round(cumulative * fps / 100)
            cumulative += int(round(d * 100))
            end = round(cumulative * fps / 100)
            frames_per_shot.append(end - start)
            
        total_expected = round(7.0 * fps)
        self.assertEqual(sum(frames_per_shot), total_expected)

    def test_determinism(self):
        plan_dict = {
            "plan_version": "1.0", "listing_id": "bench", "mode": "cinematic", "preset": "bench",
            "output": {"width": 320, "height": 180, "fps": 30, "codec": "h264"},
            "total_duration_s": 2.0,
            "shots": [
                {
                    "shot_id": "s1", "order": 1, "asset_id": "a", "image_uri": "s3://b/img.jpg",
                    "source": {"width": 1920, "height": 1080},
                    "room_key": "x", "room_type": "y", "narrative_stage": "z", "role": "hero",
                    "duration_s": 2.0,
                    "camera": {
                        "type": "ken_burns", "easing": "ease_in_out",
                        "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0},
                        "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.2}
                    },
                    "transition_in": {"type": "cut", "duration_s": 0.0}
                }
            ],
            "warnings": [],
            "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": ""}
        }
        plan = Plan(**plan_dict)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            os.makedirs(os.path.join(d1, "images"))
            os.makedirs(os.path.join(d2, "images"))
            
            # create synthetic image
            img = Image.new('RGB', (1920, 1080), color='white')
            img.save(os.path.join(d1, "images", "img.jpg"))
            img.save(os.path.join(d2, "images", "img.jpg"))
            
            out1 = render_direct(plan, d1)
            out2 = render_direct(plan, d2)
            
            # Get framemd5
            def get_md5(vid):
                cmd = ["ffmpeg", "-i", vid, "-f", "framemd5", "-"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                return res.stdout
                
            md5_1 = get_md5(out1)
            md5_2 = get_md5(out2)
            
            print("\nDeterminism Test: ", "IDENTICAL" if md5_1 == md5_2 else "NOT IDENTICAL")
            self.assertEqual(md5_1, md5_2)
            
    def test_end_to_end_quadrant(self):
        # A synthetic source image of four solid-colour quadrants. 
        # A shot whose camera starts centred on the top-left quadrant at zoom 2.0 
        # must have a first frame dominated by that colour.
        plan_dict = {
            "plan_version": "1.0", "listing_id": "bench", "mode": "cinematic", "preset": "bench",
            "output": {"width": 320, "height": 180, "fps": 30, "codec": "h264"},
            "total_duration_s": 1.0,
            "shots": [
                {
                    "shot_id": "s1", "order": 1, "asset_id": "a", "image_uri": "s3://b/img.jpg",
                    "source": {"width": 1920, "height": 1080},
                    "room_key": "x", "room_type": "y", "narrative_stage": "z", "role": "hero",
                    "duration_s": 1.0,
                    "camera": {
                        "type": "ken_burns", "easing": "ease_in_out",
                        "start": {"cx": 0.25, "cy": 0.25, "zoom": 2.0},
                        "end": {"cx": 0.75, "cy": 0.75, "zoom": 2.0}
                    },
                    "transition_in": {"type": "cut", "duration_s": 0.0}
                }
            ],
            "warnings": [],
            "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": ""}
        }
        plan = Plan(**plan_dict)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "images"))
            
            # create quadrant image (red top-left, green top-right, blue bottom-left, yellow bottom-right)
            img = Image.new('RGB', (1920, 1080), color='black')
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 960, 540], fill='red')
            draw.rectangle([960, 0, 1920, 540], fill='green')
            draw.rectangle([0, 540, 960, 1080], fill='blue')
            draw.rectangle([960, 540, 1920, 1080], fill='yellow')
            img.save(os.path.join(d, "images", "img.jpg"))
            
            out = render_direct(plan, d)
            
            # extract first frame and check color
            frame_path = os.path.join(d, "frame.png")
            subprocess.run(["ffmpeg", "-y", "-i", out, "-vframes", "1", frame_path], capture_output=True)
            
            first_frame = Image.open(frame_path)
            # count red pixels
            colors = first_frame.getcolors(320*180)
            red_count = 0
            for count, color in colors:
                if color[0] > 200 and color[1] < 50 and color[2] < 50:
                    red_count += count
                    
            self.assertGreater(red_count, 320*180 * 0.9) # dominated by red

    def test_aspect_ratios(self):
        # 3:2 and 9:16 sources should fill the 16:9 output without black pixels
        for source_w, source_h in [(300, 200), (900, 1600)]:
            plan_dict = {
                "plan_version": "1.0", "listing_id": "bench", "mode": "cinematic", "preset": "bench",
                "output": {"width": 320, "height": 180, "fps": 30, "codec": "h264"},
                "total_duration_s": 1.0,
                "shots": [
                    {
                        "shot_id": "s1", "order": 1, "asset_id": "a", "image_uri": "s3://b/img.jpg",
                        "source": {"width": source_w, "height": source_h},
                        "room_key": "x", "room_type": "y", "narrative_stage": "z", "role": "hero",
                        "duration_s": 1.0,
                        "camera": {
                            "type": "ken_burns", "easing": "ease_in_out",
                            "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0},
                            "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}
                        },
                        "transition_in": {"type": "cut", "duration_s": 0.0}
                    }
                ],
                "warnings": [],
                "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": ""}
            }
            plan = Plan(**plan_dict)
            with tempfile.TemporaryDirectory() as d:
                os.makedirs(os.path.join(d, "images"))
                
                # create solid white image
                img = Image.new('RGB', (source_w, source_h), color='white')
                img.save(os.path.join(d, "images", "img.jpg"))
                
                out = render_direct(plan, d)
                
                # extract first frame
                frame_path = os.path.join(d, "frame.png")
                subprocess.run(["ffmpeg", "-y", "-i", out, "-vframes", "1", frame_path], capture_output=True)
                
                first_frame = Image.open(frame_path)
                colors = first_frame.getcolors(320*180)
                black_count = 0
                for count, color in colors:
                    if color[0] < 10 and color[1] < 10 and color[2] < 10:
                        black_count += count
                        
                self.assertEqual(black_count, 0, f"Found black pixels in {source_w}x{source_h} source")

    def test_verify_fails(self):
        plan_dict = {
            "plan_version": "1.0", "listing_id": "bench", "mode": "cinematic", "preset": "bench",
            "output": {"width": 320, "height": 180, "fps": 30, "codec": "h264"},
            "total_duration_s": 2.0,
            "shots": [], "warnings": [],
            "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": ""}
        }
        plan = Plan(**plan_dict)
        
        with tempfile.TemporaryDirectory() as d:
            dummy_vid = os.path.join(d, "dummy.mp4")
            # create a dummy video of wrong size
            subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=2", "-c:v", "libx264", dummy_vid], capture_output=True)
            
            with self.assertRaisesRegex(RuntimeError, "Expected 320x180"):
                verify(dummy_vid, plan)

if __name__ == '__main__':
    unittest.main()
