import json
import concurrent.futures

class ModerationAgent:
    def __init__(self, client, vision_model: str, system_spine: str, few_shot_examples: str, moderation_taxonomy: str, user_prompt: str):
        self.MODEL_VISION_AGENT = vision_model
        self.client = client

        self.system_prompt = f"""
            {system_spine}
    
            TAXONOMY_JSON_START
            {moderation_taxonomy}
            TAXONOMY_JSON_END
    
            FEW-SHOT EXAMPLES START
            {few_shot_examples}
            FEW-SHOT EXAMPLES END""".strip()

        self.user_prompt = user_prompt


    def taxonomy_label_image(self, img_b64):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt},
                    {"type": "image_url", "image_url": {"url": img_b64}}
                ]
            }
        ]
        out = self.client.chat.completions.create(
            model=self.MODEL_VISION_AGENT,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        message = out.choices[0].message
        if message.refusal:
            return {
                    "taxonomy_version": "2.0",
                    "summary": "Moderation refused for safety; escalate to human.",
                    "overall_recommended_action": "BLOCK",
                    "flags": [{
                        "category_id": "E10",
                        "subcategory_id": "E10.2",
                        "rationale": "Refusal triggered.",
                        "evidence": {"type": "","pointer":""},
                        "confidence": 1.0,
                    }],
                    "text_fragments": []
                }
        if not message.content:
            return {
                    "taxonomy_version": "2.0",
                    "summary": "Moderation failed for unknown reasons.",
                    "overall_recommended_action": "REQUIRE_EDITS",
                    "flags": [{
                        "category_id": "E10",
                        "subcategory_id": "E10.1",
                        "rationale": "No moderation results available.",
                        "evidence": {"type": "","pointer":""},
                        "confidence": 1.0,
                    }],
                    "text_fragments": []
                }

        return json.loads(out.choices[0].message.content)



    def process_image(self, _img_64, name):
        response = self.taxonomy_label_image(_img_64)
        return (
            name,
            response
        )

    def moderate_images(self, image_dict: dict):
        taxonomy_moderation_results = {}


        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_image, image, name) for name, image in image_dict.items()]

            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                taxonomy_moderation_results[name] = result
                print("completed", name)

        return taxonomy_moderation_results