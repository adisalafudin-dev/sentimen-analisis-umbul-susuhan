import pandas as pd
import matplotlib.pyplot as plt


# Diagram batang awal

def main(): 
    df = pd.read_csv("processed_data.csv")

    labels = ['Positive', 'Negative']
    data_count = [df['status'].value_counts()['positive'], 
                df['status'].value_counts()['negative']]

    plt.bar(labels, data_count, tick_label=labels, width=0.5)
    plt.xlabel("Kelas Sentimen")
    plt.ylabel("Data")
    plt.title("Diagram Bar Data Analisis Sentimen")

    plt.show()


if __name__ == "__main__":
    main()