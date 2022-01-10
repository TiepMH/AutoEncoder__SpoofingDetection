import numpy as np
from library.class_AE import get_compiled_model


def evaluate_model_and_get_decoded_imgs(imgs_train_normal,
                                        imgs_test_normal,
                                        imgs_test_anomalous,
                                        n_angles,
                                        n_epochs):
    # imgs_test = np.vstack((imgs_test_normal,imgs_test_anomalous))
    model = get_compiled_model(n_angles)
    history = model.fit(imgs_train_normal,
                        imgs_train_normal,
                        epochs=n_epochs, batch_size=2**6,
                        validation_data=(imgs_test_normal, imgs_test_normal),
                        shuffle=True, verbose=False)
    # evaluate the model
    loss_train = history.history['loss']
    loss_test = history.history['val_loss']
    #
    imgs_train_normal_encoded = model.encoder(imgs_train_normal).numpy()
    imgs_train_normal_decoded = model.decoder(imgs_train_normal_encoded).numpy()
    imgs_test_normal_encoded = model.encoder(imgs_test_normal).numpy()
    imgs_test_normal_decoded = model.decoder(imgs_test_normal_encoded).numpy()
    imgs_test_anomalous_encoded = model.encoder(imgs_test_anomalous).numpy()
    imgs_test_anomalous_decoded = model.decoder(imgs_test_anomalous_encoded).numpy()
    return model, history, loss_train, loss_test, \
        imgs_train_normal_decoded, \
        imgs_test_normal_decoded, \
        imgs_test_anomalous_decoded
