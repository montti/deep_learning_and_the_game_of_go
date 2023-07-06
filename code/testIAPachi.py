from dlgo.gtp.play_local import LocalGtpBot
from dlgo.agent.termination import PassWhenOpponentPasses
from tensorflow import keras
from dlgo.agent.predict import DeepLearningAgent
from dlgo.encoders.oneplane import OnePlaneEncoder

encoder = OnePlaneEncoder((9, 9))

model = keras.models.load_model("deep7.h5")

deep = DeepLearningAgent(model, encoder)


gtp_bot = LocalGtpBot(go_bot=deep, termination=PassWhenOpponentPasses(), handicap=0, opponent='gnugo', our_color='w')
gtp_bot.run()
