from tensorflow import keras
from dlgo.agent.predict import DeepLearningAgent
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo import httpfrontend
from dlgo import mcts


encoder = OnePlaneEncoder((9, 9))

model = keras.models.load_model("deep7.h5")

deep = DeepLearningAgent(model, encoder)

web_app = httpfrontend.get_web_app({'deep': deep})
web_app.run()

