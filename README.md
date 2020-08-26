# Vowel Finder 

## Description
This was my final project at Flatiron Data Science Bootcamp. Here, **I aim to find all acoustic vowels in an unmarked audio file of natural conversational speech**. By doing so, one would be able to use this as a basis for an Automatic Speech Recognition system and could be used to identify languages or be used in phonetic research. 

The data originally came from a [study](http://groups.linguistics.northwestern.edu/speech_comm_group/wildcat/) done by Northwestern University Linguistics Department. The [transcriptions](https://speechbox.linguistics.northwestern.edu/wildcat/#!/recordings) were auto-aligned from transcriptions and hand corrected by the studies' authors. Unfortunately, the quality of the annotations were not sufficient, so I corrected one, 10-minute interview. The reason corrections were needed is that they were annotated at the _word_ level, and not the _acoustic_ level. For example, the first _e_ in "interesting" is often acoustically absent (there is a "chr" sound, not "ter"). In the original transcription the "e" sound will be marked, even if not actually present.

I investigate the effectiveness of four different approaches in this project: 
- **[Hampel Filter](https://dsp.stackexchange.com/questions/26552/what-is-a-hampel-filter-and-how-does-it-work)**: One finds the median volume of a small section of the audio file, and determines if the sample in question is above a threshold of standard deviations of that median volume. 
- **Neural Network**: Feeds the model frames of vowels and performs a 2D vanilla neural network to find vowels in the test data.
- **Combination Model**: This model combines the results of the previous two models to hopefully obtain more accurate predictions.
- **Ensemble Learning Algorithm (Random Forest and Bagging Classifier)**: These models use decision trees on samples of the data and then uses the values that distinguish those data points on the rest of the data. The difference between the two is that of all the features that are found, the Random Forest algorithm considers all in tandem whereas the Bagging Classifier considers each one individually, and then uses majority voting to determine the final prediction. 

## Results
The Neural Network model and Combination model both performed equally well. The Hampel Filter had a significantly lower effectiveness than the other two approaches. Since there is little if any gain from using a combination model, a 2D neural network is the best approach used here.

The results are presented in a [deck](https://docs.google.com/presentation/d/1E3h6cjbQvKEGE3kDF8XS36_IW0HKgu7hBuGMthUwPsc/edit?usp=sharing) that was presented as part of the Flatiron Bootcamp's Science Fair.

## Technologies Used
- Jupyter Notebook
- Python
    - Pandas
    - os
    - re
    - SciPy
    - NumPy
    - LibROSA 
    - Textgrids
    - numba
    - Seaborn
    - Matplotlib
    - Plotly
    - Keras
    - scikit-learn

## Future Directions
This project was a first attempt at finding the most successful approach to finding a specific class of speech sounds acoustically. There are two areas of improvement: the phonetic side, and the data science side.

On the phonetic side, one could take more time to identify the main characteristics of vowels as compared to consonants. For example, vowels can carry pitch, so by measuring that aspect of a suspected vowel, we can be more sure we are correctly identifying vowels. Additionally, we could look at voicing, or more specifically, energy in the 50-100 Hz range, to better identify vowels.

On the data Science side, only a 2D neural network was found to have any significant result (many approaches had guessed the dominant class for all examples given). If we adjusted the type of neural network used and the parameters used with the neural network, we may have been more successful in finding vowels. 

In the future, I suggest using a neural network approach to finding vowels. While the Hampel Filter has the potential to perform very well with the correct sigma and filter window (the only two parameters it has), the time needed to find these parameters is likely much longer and requires more computation than to to use a neural network. 
