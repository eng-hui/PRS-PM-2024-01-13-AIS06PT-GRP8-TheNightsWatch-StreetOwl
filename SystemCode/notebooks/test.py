


#####OWL 
import requests
from PIL import Image
import torch
import supervision as sv
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import scipy

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-finetuned")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-finetuned")

file = "./SystemCode/notebooks/test.jpg"
image = Image.open(file)
texts = [["an old lady"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.3)
# i = 0  # Retrieve predictions for the first image for the corresponding text queries
# text = texts[i]
# boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
# for box, score, label in zip(boxes, scores, labels):
#     box = [round(i, 2) for i in box.tolist()]
#     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
# for result in results:
#     det = sv.Detections.from_transformers(result)
detection = results[0]
det = sv.Detections.from_transformers(detection)
box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(
  scene=image.copy(),
  detections=det)

from matplotlib import pyplot as plt
plt.imshow(annotated_frame)
plt.show()