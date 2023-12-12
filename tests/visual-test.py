import matplotlib.pyplot as plt
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()

model_names = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
    "SFace",
]
detector_backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]

img_path_1 = "/Users/stevennguyen/Documents/VNEXT/VNEXT-Projects/VNEXT-FaceRecognition/deepface/tests/dataset/img1.jpg"
img_path_2 = "/Users/stevennguyen/Documents/VNEXT/VNEXT-Projects/VNEXT-FaceRecognition/deepface/tests/dataset/img2.jpg"

# verification
for model_name in model_names:
    obj = DeepFace.verify(
        img1_path=img_path_1, img2_path=img_path_2, model_name=model_name
    )
    logger.info(obj)
    logger.info("---------------------")

# represent
for model_name in model_names:
    embedding_objs = DeepFace.represent(img_path=img_path_1, model_name=model_name)
    for embedding_obj in embedding_objs:
        embedding = embedding_obj["embedding"]
        logger.info(f"{model_name} produced {len(embedding)}D vector")

# find
dfs = DeepFace.find(
    img_path=img_path_1, db_path="tests/dataset", model_name="Facenet", detector_backend="mtcnn"
)
for df in dfs:
    logger.info(df)

# extract faces
for detector_backend in detector_backends:
    face_objs = DeepFace.extract_faces(
        img_path=img_path_1, detector_backend=detector_backend
    )
    for face_obj in face_objs:
        face = face_obj["face"]
        logger.info(detector_backend)
        plt.imshow(face)
        plt.axis("off")
        plt.show()
        logger.info("-----------")
