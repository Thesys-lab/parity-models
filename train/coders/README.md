# Coders
This directory contains implementations of encoders and decoders.
All encoders and decoders derive from a common [Coder](coder.py) class.

### Adding a new encoder/decoder
To add a new encoder/decoder, implement a class that derives from [Coder](coder.py),
and add the classpath of your encoder/decoder to [train_config.py](../train_config.py)
under `get_encoder` and `get_decoder`.
