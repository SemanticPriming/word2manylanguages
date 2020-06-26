The Multilanguage Written Picture Naming Dataset is described in 

Torrance, M., Nottbusch, G., Alves, R. A., Arfé, B., Chanquoy, L., Chukharev-Hudilainen, E., … Wengelin, Å. (in press). Timed Written Picture Naming in 14 European Languages. Behavior Research Methods.

We provide two files giving timing norms: MWPND_by_picture.txt and MWPND_by_trial.txt. Both are tab-delimited text files with one variable per column. Both files contain text in various alphabets and are UTF8 encoded. Variable labels are given in a header row and briefly explained below. Please see the paper for detailed descriptions of the variables.

The by-keypress data from which these were derived is also potentially available. If you would like to make use of part or all of these data, please contact mark.torrance@ntu.ac.uk  / m.s.torrance@gmail.com

MWPND_participants.txt gives age, sex and self-reported typing ability. See below. 

The Rossion and Pourtois coloured picture set is available online in various easy-to-google places.

MWPND_by_picture.txt gives the following variables...

language
picture identifier
most common response
proportion of participants giving most common response
proportion of participants giving most common name
name H (see paper)
proportion of the subjects who gave most common name that also gave this name with its most common spelling
spell H (see paper)
proportion null responses
mean onset latency, all names (see paper for screening details)
SD of onset latency, all names
response latency upper bound on lower quartile, all names
median response latency, all names 
response latency lower bound of upper quartile, all names
mean of mean inter-keypress interval, all names
SD of mean inter-keypress interval, all names
mean inter-keypress interval upper bound on lower quartile, all names
median of mean inter-keypress interval, all names
mean inter-keypress interval lower bound of upper quartile, all names
mean onset latency, subjects giving most common name (see paper for screening details)
SD of onset latency, subjects giving most common name
response latency upper bound on lower quartile, subjects giving most common name
median response latency, subjects giving most common name 
response latency lower bound of upper quartile, subjects giving most common name
mean of mean inter-keypress interval, subjects giving most common name
SD of mean inter-keypress interval, subjects giving most common name
mean inter-keypress interval upper bound on lower quartile, subjects giving most common name
median of mean inter-keypress interval, subjects giving most common name
mean inter-keypress interval lower bound of upper quartile, subjects giving most common name
alternative names (name_number of subjects giving this name_proportion of subjects) - up to maximum of 5
alternative spellings of most common name (spelling _ number of subjects giving this name _ ratio relative to most common response _ levenshtein ratio (relative to most common response)_levenshtein distance) - up to maximum of 5
image file size (pixels)
mean image familiarity rating (from Rossion and Pourtois, 2004)
mean image complexity rating (from Rossion and Pourtois, 2004)
mean rating of match of picture to mental image (from Rossion and Pourtois, 2004)
length of most common response (letters)
length of most common response (words)


MWPND_by_trial.txt gives the following variables...
lang	language
subno	subject number
image	picture identifier
response	response
is_fluent	1 = response produced without editing
is_mc_response	1 = response is the most common response
is_mc_name	1 = response respresents the most common name
RT	onset latency
MIKI	mean inter-keypress interval
response_length_chars	number of characters in response
response_length_words	number of orthographic words in response
lev_ratio	Levenshtein ratio, response vs. most common response (only meaningful for trials where is_mc_name = 1)
lev_distance	Levenshtein distance, response vs. most common response (only meaningful for trials where is_mc_name = 1)

MWPND_participants gives the following variables...
lang	language
subno	subject number
sex	1 = male, 2 = female
age	in years, derived from date of birth
typing_ability	self reported typing ability: How strong would you say your typing ability is? 
	1 to 7 with 1 = very good, 4 = moderate, 7 = very poor
typing_method	Which of the following options best describes your method of typing?
	1 = I use only one index finger (first finger).
	2 = I use the index and middle finger of one hand.
	3 = I use the index fingers of both hands.
	4 = I use the index and middle finger of both hands.
	5 = I use all fingers of one hand.
	6 = I use all fingers of both hands.
	Other (recoded into one of the above categories)
typing_gaze	When typing, where do you mainly focus your eyes?
	1 = Always at the computer screen.
	2 = Always at the keyboard.
	3 = Mainly at the computer screen.
	4 = Mainly at the keyboard.
	5 = About half and half at the computer screen and the keyboard.
	Other (recoded into one of the above categories)


