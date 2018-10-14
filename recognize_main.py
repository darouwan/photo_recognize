import sys
from model_util import *

if __name__ == "__main__":
    training_path = sys.argv[1]
    test_path = sys.argv[2]

    print("Training KNN classifier...")
    classifier = train(training_path, model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir(test_path):
        full_file_path = os.path.join(test_path, image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join(test_path, image_file), predictions)
