import asyncio
import cv2
import numpy as np
from app.database import init_db, AsyncSessionLocal
from app.api.routes import _run_pipeline

async def test():
    await init_db()
    # Create synthetic test page image (white background with black text)
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Student Handwriting Sample", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Line two of the test assignment", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    async with AsyncSessionLocal() as db:
        doc, nodes = await _run_pipeline(db, img, "test_sample.png", "STU101", "Alice", "7A")
        print(f"PIPELINE SUCCESS! Doc ID: {doc.id}, Status: {doc.status}, Nodes count: {len(nodes)}")
        for n in nodes:
            print(f"  Node: raw=\"{n.raw_text}\", corrected=\"{n.corrected_text}\", conf={n.confidence_score:.3f}")

if __name__ == "__main__":
    asyncio.run(test())
