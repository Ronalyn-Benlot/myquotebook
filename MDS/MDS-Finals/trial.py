import numpy as np
import cv2
from timeit import default_timer as timer



# This class represents a character in a word
class WordCharacter:
    def __init__(self, code, path):
        self.char_code = code
        self.img_path = path

    def __str__(self):
        return f'char_code: {self.char_code}, img_path: {self.img_path}'

# This class is used to translate a word from an image to a string
class Translator:

    RESIZED_IMAGE_WIDTH = 30
    RESIZED_IMAGE_HEIGHT = 40
    FLATTENED_IMGS = None
    INT_CLASSIFICATIONS = None
    CLASSIFICATION_FILE = "classifications.txt"
    FLAT_CHAR_IMGS_FILE = "flatCharImages.txt"
    WORD_CHARACTERS = None

    def __init__(self, word_characters: WordCharacter = None):
        self.WORD_CHARACTERS = word_characters if word_characters is not None else dict()
        self.INT_CLASSIFICATIONS = []
        self.FLATTENED_IMGS = np.empty((0, self.RESIZED_IMAGE_WIDTH * self.RESIZED_IMAGE_HEIGHT))
        self.KNN = cv2.ml.KNearest_create()

    def add_character(self, word_character: WordCharacter):
        is_number = word_character.char_code >= 48 and word_character.char_code <= 57
        is_uppercase = word_character.char_code >= 65 and word_character.char_code <= 90
        is_lowercase = word_character.char_code >= 97 and word_character.char_code <= 122
        if not is_number and not is_uppercase and not is_lowercase:
            raise ValueError(f'Character code is not valid: {word_character.char_code}')
        if word_character.char_code in self.WORD_CHARACTERS.keys():
            raise ValueError("Character code already exists")

        self.WORD_CHARACTERS[word_character.char_code] = word_character.img_path
    
    def update_character(self, word_character: WordCharacter):
        if word_character.char_code not in range(0, 9+1) and word_character.char_code not in range(65, 90+1) and word_character.char_code not in range(97, 122+1):
            raise ValueError(f'Character code is not valid: {word_character.char_code}')
        if word_character.char_code not in self.WORD_CHARACTERS.keys():
            raise ValueError("Character code does not exist")

        self.WORD_CHARACTERS[word_character.char_code] = word_character.img_path
    
    
    def generate_train_data(self, classifications_file_name: str = None, flattened_images_file_name: str = None):
        if classifications_file_name is not None:
            self.CLASSIFICATION_FILE = classifications_file_name
        if flattened_images_file_name is not None:
            self.FLAT_CHAR_IMGS_FILE = flattened_images_file_name

        print(f'words: {self.WORD_CHARACTERS.keys()}')

        for char_code, img_path in self.WORD_CHARACTERS.items():
            img_train = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY)
            img_tresh = cv2.threshold(img_gray, 150, 255, cv2.CHAIN_APPROX_NONE)[1].copy()
            
            img_contours = cv2.findContours(img_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

         
            for contour in img_contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                img_roi = img_tresh[y:y+h, x:x+w]
                img_resized_roi = cv2.resize(img_roi, (self.RESIZED_IMAGE_WIDTH, self.RESIZED_IMAGE_HEIGHT))
                self.INT_CLASSIFICATIONS.append(char_code)
                flattened_img = img_resized_roi.reshape((1, self.RESIZED_IMAGE_WIDTH * self.RESIZED_IMAGE_HEIGHT))
                self.FLATTENED_IMGS = np.append(self.FLATTENED_IMGS, flattened_img, 0)

            print(f'char_code: {char_code}({chr(char_code)}), img_path: {img_path}, contour len: {len(img_contours)}')
                
        flattened_classifications = np.array(self.INT_CLASSIFICATIONS, np.float)
       
        final_classifications = flattened_classifications.reshape((flattened_classifications.size, 1))

        np.savetxt(self.CLASSIFICATION_FILE, final_classifications)
        np.savetxt(self.FLAT_CHAR_IMGS_FILE, self.FLATTENED_IMGS)
            

    def train(self, train_img_path: str):
        char_classifications = np.loadtxt(self.CLASSIFICATION_FILE, np.float32)
        flat_char_images = np.loadtxt(self.FLAT_CHAR_IMGS_FILE, np.float32)
        char_classifications = char_classifications.reshape((char_classifications.size, 1))

        self.KNN.train(flat_char_images, cv2.ml.ROW_SAMPLE, char_classifications)
        img_sample = cv2.imread(train_img_path)
        start = timer()
       
        # bf = cv2.bilateralFilter(img_sample, 15, 75, 75)
        bf = cv2.bilateralFilter(img_sample, 50, 100, 100)
        end = timer()
        print(f'b filter: {end-start}s')
        
        grayscale_img = cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY)
        img_tresh = cv2.threshold(grayscale_img, 150, 255, cv2.CHAIN_APPROX_NONE)[1].copy()
        

        img_contours = list(cv2.findContours(img_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]).copy()
      
        c = 0
        predicted_str = ""

        print(f'contours len: {len(img_contours)}')
        for cntrs in img_contours:
            approximation = cv2.approxPolyDP(cntrs, 0.01 * cv2.arcLength(cntrs, True), True)
           
            if len(approximation) == 4:
                if len(str(cv2.contourArea(approximation))) > 6:
                    pass
                else:
                    [intX, intY, intW, intH] = cv2.boundingRect(approximation)
                    cv2.rectangle(img_sample, (intX, intY), (intX + intW, intY + intH), (0, 255, 0), 2)

                    img_char = img_tresh[intY:intY + intH, intX:intX + intW]
                    img_char = img_char[5:50, 5:48] #remove border
                    cv2.imshow("img_sample", img_sample)
                    cv2.waitKey(0)

                    inverted_img = cv2.bitwise_not(img_char)
                    retval, img_tresh2 = cv2.threshold(inverted_img, 150, 255, cv2.CHAIN_APPROX_NONE)
                    contours, h = cv2.findContours(img_tresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        c+=1
                        if c%2 != 1:
                            continue
                        [x, y, w, h] = cv2.boundingRect(contour)
                        img_roi = img_tresh2[y:y + h, x:x + w]
                        img_resized_roi = cv2.resize(img_roi, (self.RESIZED_IMAGE_WIDTH, self.RESIZED_IMAGE_HEIGHT))
                        
                        flattened_img = img_resized_roi.reshape((1, self.RESIZED_IMAGE_WIDTH * self.RESIZED_IMAGE_HEIGHT))
                        flattened_img = np.float32(flattened_img)
                        
                        retval, results, neigh_resp, dists = self.KNN.findNearest(flattened_img, k=1)
                        string_char = str(chr(int(results[0][0])))
                        predicted_str = predicted_str + string_char

                        # cv2.imshow("img_resized_roi", img_resized_roi)
                        # cv2.waitKey(0)
        

        print(f'predicted_str({len(predicted_str)}): "{predicted_str}"')
    
    
    def __str__(self):
        return f'WORD_CHARACTERS: {self.WORD_CHARACTERS}'
