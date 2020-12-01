import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, matrix, histogram):
        self.matrix = matrix
        self.histogram = histogram

    
    def get_matrix(self):
        return self.matrix

    def get_histogram(self):
        return self.histogram

    def get_matrix_len(self):
        return len(self.matrix)

    def plot(self):
        fig = plt.figure(figsize=(20, 20))

        if not self.histogram:

            for i in range(len(self.matrix)):

                ax = fig.add_subplot(2,2,i+1)
                ax.imshow(self.matrix[i][0], interpolation='none', cmap='gray')
                ax.set_title(self.matrix[i][1])
        else:
            ax = fig.add_subplot(1,2,1)
            ax.imshow(self.matrix[0], interpolation = 'none', cmap = 'grey')

            ax1 = fig.add_subplot(1, 2, 1)

            ax1.hist(self.matrix[0].ravel(), bins=256)
            ax1.set_title('Histogram')  # This is our histogram
        plt.show()