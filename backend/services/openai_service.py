import base64
from openai import AsyncOpenAI

from config import OPENAI_API_KEY

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def generate_thumbnail(prompt: str, style_prompt: str, headshot_url: str)-> bytes:
    """
    USe the Response API with gtp-image-2 as a built-in image_generation tool.
    Pass the headshot URL directly as an input_image.
    Return raw PNG bytes.
    """

    full_prompt = (
        f"{style_prompt}\n\n"
        f"User request: {prompt}\n\n"
        "IMPORTANT: The generated thumbnail must prominently feature the person shown in the provided reference headshot photo. Keep their likeness accurate."
    )

    response = await client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_image", "image_url": headshot_url},
                    { "type": "input_text", "text": full_prompt}
                ]
            }
        ],
        tools=[
            {
                "type": "image_generation",
                "model": "gpt-image-2",
                "size": "1536x1024",
                "quality": "medium",
                "output_format": "png"
            },
            
        ]
    )

    for item in response.output:
        if item.type == "image_generation_call" and item.result:
            return base64.b64decode(item.result)
    
    raise RuntimeError("No image generation result found in the response")

# import google.generativeai as genai
# from config import GEMINI_API_KEY

# genai.configure(api_key=GEMINI_API_KEY)

# model = genai.GenerativeModel("gemini-2.5-flash-image")

# async def generate_thumbnail(prompt, style_prompt, headshot_url):

#     full_prompt = f"""
#     {style_prompt}

#     User request:
#     {prompt}
#     """

#     response = model.generate_content([
#         full_prompt,
#         {
#             "file_data": {
#                 "mime_type": "image/jpeg",
#                 "file_uri": headshot_url
#             }
#         }
#     ])

#     return response.candidates[0].content.parts[0].inline_data.data