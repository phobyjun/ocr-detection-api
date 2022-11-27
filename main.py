from fastapi import FastAPI, HTTPException
from mangum import Mangum

from app.textract import get_date_from_image, NoDateInImageException

app = FastAPI()


@app.get("/ocr/health-check", status_code=200)
async def health_check():
    return {"status": "success"}


@app.get("/ocr/{image_url}")
async def ocr_detection_api(image_url: str):
    try:
        expiration_date = get_date_from_image(image_url)
        return {
            "image_url": image_url,
            "expiration_date": expiration_date
        }
    except NoDateInImageException as e:
        raise HTTPException(status_code=404, detail=e.__str__())
    except Exception as e:
        raise HTTPException(status_code=500, detail=e.__str__())


handler = Mangum(app)
