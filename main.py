import cv2
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st
import PIL
from sklearn.neighbors import NearestNeighbors


def trainModel():
    descriptorsList = []
    resultDescriptorsList = []

    print("Getting image descriptors...")
    for file in tqdm(glob("train2017/*")):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        descriptors = sift.detectAndCompute(img, None)[1]
        descriptorsList += list(descriptors)

    resultDescriptorsList = np.array(descriptorsList)

    print("Training model on descriptors...")
    model = KMeans(n_clusters=2048)
    model.fit(resultDescriptorsList)

    print("Saving model to pickle file...")
    with open("trainedModel.pickle", "wb") as modelFile:
        pickle.dump(model, modelFile)
    print("Sucessfully created trained model !")


def imgVec(img, model):
    sift = cv2.SIFT_create()
    descriptors = sift.detectAndCompute(img, None)[1]
    descClass = model.predict(descriptors)
    hist = np.histogram(descClass, 2048, density=True)[0]
    return hist


def createDatabase():
    with open("trainedModel.pickle", "rb") as modelFile:
        model = pickle.load(modelFile)
    imgPaths = []
    imgVectors = []
    print("Pushing images to vectors...")
    for file in tqdm(glob("testData/*")):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        imgVector = imgVec(img, model)
        imgPaths.append(file)
        imgVectors.append(imgVector)
    print("Creating database file...")
    database = pd.DataFrame({"path": imgPaths, "vector": imgVectors})
    database.to_pickle("database.pickle")
    print("Sucessfully created database file !")


@st.cache(allow_output_mutation=True)
def getNeighbours(database):
    neighbours = NearestNeighbors(n_neighbors=5, metric="cosine")
    neighbours.fit(np.stack(database["vector"].to_numpy()))
    return neighbours


# trainModel()
# createDatabase()


@st.cache(allow_output_mutation=True)
def getData():
    database = pd.read_pickle("database.pickle")
    with open("trainedModel.pickle", "rb") as modelFile:
        model = pickle.load(modelFile)
    neighbours = getNeighbours(database)
    return database, model, neighbours


database, model, neighbours = getData()
st.title("Search for similar images")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
if uploaded_file is not None:
    imgBufferArr = np.frombuffer(uploaded_file.getbuffer(), dtype="uint8")
    img = cv2.imdecode(imgBufferArr, cv2.IMREAD_GRAYSCALE)
    imgVector = imgVec(img, model)

    neighboursIndices = neighbours.kneighbors(imgVector.reshape(1, -1), return_distance=False)[0]
    similarImgPaths = np.hstack(database.loc[neighboursIndices, ["path"]].values)

    for imgPath in similarImgPaths:
        st.image(imgPath, caption=imgPath)


