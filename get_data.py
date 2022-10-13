class DataReader():
    """
    Get dataset from files

    Examples:
        train, test = DataReader("data/train.txt","data/test.txt").read()
    """
    def __init__(self, train_file, test_file):
        """
        Init dataset information.

        Inputs:
            train_file: train file's location & full name
            test_file: test file's location & full name

        Examples:
            DataReader("data/train.txt","data/test.txt")
        """
        self.train_file = train_file
        self.test_file = test_file
        self.tarin_raw = []
        self.test_raw = []

    def get_raw(self, input_file):
        """
        Get raw data from file

        Inputs:
            input_file: input file name

        Returns:
            raw_data: a set with raw data

        Examples:
            raw = get_raw("data/train.txt")
        """
        with open(input_file) as reader:
            raw_data = reader.readlines()
        
        return raw_data

    def formate(self, raw_data):
        """
        Formate raw data

        Inputs:
            raw_data: a set with raw data

        Returns:
            dataset: a set with formated data

        Examples:
            raw = ["1 Abc def\\n", "0 xyz"]
            dataset = formate(raw)
            assert(dataset == [(1, "abc def"]), (0, "xyz")])
        """
        dataset = []

        for raw in raw_data:
            num_idx = 0
            while raw[num_idx] not in "0123456789":
                num_idx += 1
            
            label = int(raw[ : num_idx + 1])

            str_idx = num_idx + 1
            if raw[str_idx] == ' ':
                str_idx += 1

            if raw[-1] == "\n":
                string = raw[str_idx : -1]
            else:
                string = raw[str_idx : ]

            string.lower()

            dataset.append((label, string))

        return dataset

    def read(self):
        """
        Get dataset and formate.

        Returns:
            train: train dataset
            test: test dataset

        Examples:
            train, test = read()
        """
        train = self.formate(self.get_raw(self.train_file))
        test = self.formate(self.get_raw(self.test_file))

        return train, test
