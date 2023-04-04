# NLP Assignments
 
mood_lyric_generator tokenizes each words including "<end of line>", "<end of lyric>" , "<comma>" and "<period>" of lyrics of 57650 songs.

There are 4 features shown in the terminal when you run the code:
1. There are 4 examples of song. Each example shows the list of words for the title and another list of words for the lyrics.
2. Next, there are 5 special cases: P(A | B, C). This shows the probability of the word A coming out when the two preceding words were B and C. 
For example, P(you | I, love) means the probability of the word "you" coming out after "I love" which is around 4.7%.
3. The third feature shows list of the songs that was associated with a given word. Notice how there are no songs associated with "red" and "blue". This is because there were small number of cases where those two words were used.
4. Finally, the last feature shows a song lyric generated based on a keyword(mood). 
