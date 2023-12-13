import numpy as np

class OneHotEncoder:
    """
    One-hot encoding is a representation technique where categorical data, such as words in a text sequence (or
    characters in a sequence), is converted into binary vectors with only one element set to 1 (indicating the
    presence of a specific category) and the rest set to 0.
    """

    def __init__(self, padder:str, max_length: int = None):
        """
        Parameters
        ----------
        padder:
            character to perform padding with
        max_length:
            maximum length of sequences
        
        Attributes
        -----------
        alphabet:
            the unique characters in the sequences
        char_to_index:dict
            dictionary mapping characters in the alphabet to unique integers
        index_to_char:dict
            reverse of char_to_index (dictionary mapping integers to characters)
        """
        # arguments
        self.padder = padder
        self.max_length = max_length


        #estimated parameters
        self.alphabet=set()
        self.char_to_index={}
        self.index_to_char={}
    
    def fit (self, data: list[str])->'OneHotEncoder':
        """
        Fits to the dataset.

        Parameters
        ---------
        data:list[str]
            list of sequences to learn from
        Returns
        -------

        """
        
        if self.max_length is None:
            lengths = []
            for sequence in data:
                lengh = len(sequence)
                lengths.append(lengh)
            self.max_length = np.max(lengths) #max length in data

        # pad the sequences with the padding character

        all_seq= "".join(data) # aggregate all seq to find the different characters
        self.alphabet = np.unique(list(all_seq)) # finding unique characters - getting a list

        
        indexes = np.arange(1, len(self.alphabet) + 1) # create an array of indexes from 1 to the length of the alphabet
        
        self.char_to_index = dict(zip(self.alphabet, indexes)) # create a dictionary that maps each character in the alphabet to its corresponding index
        self.index_to_char = dict(zip(indexes,self.alphabet)) # create a dictionary that maps each index to its corresponding character in the alphabet (i.e: {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'})


        #checking whether a special padding character is present in the alphabet
        if self.padder not in self.alphabet:
            self.alphabet = np.append(self.alphabet, self.padder)  #ff the padding character is not in the alphabet, it appends the padding character to the end of the alphabet
            max_index = max(self.char_to_index.values())  # finds the maximum index value currently present in the char_to_index dictionary
            new_index = max_index + 1  # calculates a new index value for the padding character by incrementing the maximum index value
            self.char_to_index[self.padder] = new_index  # adds an entry to the char_to_index dictionary, mapping the padding character to its new index value
            self.index_to_char[new_index] = self.padder # adds an entry to the index_to_char dictionary, mapping the new index value to the padding character
        
                
        return self
    
    def transform(self, data:list[str]) ->np.ndarray:
        """
        Parameter
        ---------
        data:list[str]
            data to encode
        
        Returns
        --------
        np.ndarray:
            One-hot encoded matrices
        """
        # trim each sequence to the maximum length and pad with the specified character
        sequence_trim_pad = []
        for sequence in data:
            trim_pad = sequence[:self.max_length].ljust(self.max_length, self.padder)
            sequence_trim_pad.append(trim_pad)
        #ljust -> left-justifies a string within a specified width by padding it with a specified character (or whitespace by default) on the right side 
        
        
        sorted_alphabet = sorted(self.alphabet)  # sort the classes to align with scikit-learn's lexicographic order
        class_indices = [self.char_to_index[char] for char in sorted_alphabet]

        # Create a new identity matrix with sorted classes
        identity_matrix = np.eye(len(sorted_alphabet))

        one_hot_encode = []
        for adjusted_seq in sequence_trim_pad:
            for letter in adjusted_seq:
                if letter != adjusted_seq:
                    value_in_dict = self.char_to_index.get(letter)
                    one_hot_sequence = identity_matrix[class_indices.index(value_in_dict)]
                    one_hot_encode.append(one_hot_sequence)

        return one_hot_encode

    def fit_transform(self, data: list[str]) -> np.ndarray:
        """
        Parameters
        ----------
        data: list[str]
            list of sequences to learn from
        Returns
        -------
        np.ndarray:
            One-hot encoded matrices
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> list[str]:
        """
        Parameters
        ----------
        data: np.ndarray-vem de cima
            one-hot encoded matrices to decode
        Returns
        -------
        list[str]:
            Decoded sequences
        """
        
        index = []
        for one_hot_matrix in data:
            indexes = np.argmax(one_hot_matrix)  #finding index of the max value (1)
            index.append(indexes)  #appended to list, where those indexes represent the position where the original characters were encoded

        total_sequences = []
        for each_index in index:
            char = self.index_to_char.get(each_index + 1)  #retrieves the corresponding character from the dict
            total_sequences.append(char)
            text ="".join(total_sequences) # join all chars into one string
        
        trimmed_segments = []
        for i in range (0,len(text),self.max_length): #analyses each piece at time
            string = text[i:i + self.max_length]
            trimmed_string = string.rstrip(self.padder) #trims the occurences of the padding character 
            trimmed_segments.append(trimmed_string)
        return trimmed_segments #decoded sequences

#testing

if __name__ == '__main__':
    from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

    encoder = OneHotEncoder(padder="?", max_length=9)
    data = ["abc", "aabd"]
    encoded_data = encoder.fit_transform(data)
    print("My One-Hot Encoding:")
    print(np.array(encoded_data))
    decoded_data = encoder.inverse_transform(encoded_data)

    print("\nDecoded Data:")
    print(decoded_data)

    sklearn_encoder = SklearnOneHotEncoder(sparse = False, handle_unknown = 'ignore')
    sklearn_data = np.array(data).reshape(-1, 1)
    sklearn_encoded_data = sklearn_encoder.fit_transform(sklearn_data)
    
    #print("\nScikit-learn One-Hot Encoding:")
    #print(sklearn_encoded_data)

    sklearn_decoded_data = sklearn_encoder.inverse_transform(sklearn_encoded_data)
    print("\nScikit-learn Decoded Data:")
    print(sklearn_decoded_data)