from multiprocessing.managers import SyncManager

class ImageManager(SyncManager):
    """
    Controls access to shared dict object

    get_dict() returns the Manager.dict()
    Key     Value
    camera  Camera object (webcam image)
    state   status of tracker (string)
    encoded encoded camera image with overlay (string)
    """
    pass

if __name__ == "__main__":
    # Shared Manager object
    image_dictionary = Manager().dict()
    ImageManager.register('get_dict', callable=lambda:image_dictionary)
    manager = ImageManager(address=('', 11579), authkey=b'password')
    server = manager.get_server()
    server.serve_forever()