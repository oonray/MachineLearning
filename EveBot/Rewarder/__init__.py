import numpy as np
import requests
from mss import mss
import eveapi.build.lib.eveapi as eveapi
import pytesseract as tes
import re
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

auth_url = "https://login.eveonline.com/oauth/token"
c_id = "074119f2f715412893265140ae6ea484"
secret = 'PGCMFWWIYicaMPGvT0237mKs4J3Fns7iJOPqsdS2'

requests.post(
    auth_url,

)


