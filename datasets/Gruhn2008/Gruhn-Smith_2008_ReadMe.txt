Gruhn_ReadMe.txt

//------
This ReadMe file contains descriptions of variables included in the AGE word dataset, Gruhn-Smith_2008_AGE.txt. 

For each of the 200 adjectives, the dataset contains a broad spectrum of information. The dataset includes (a) objective characteristics (word length and word frequency), (b) subjective ratings from Study I and Study II, and (c) analyses of variance testing age-related differences in the subjective evaluations. The subjective ratings include valence, arousal, control, imagery, self-relevance, and age-relevance from Study I and subjective ratings of self-other relevance by graduate psychology students from Study II. 

In order to avoid problems with character encoding, German umlauts and the German Eszett were transformed to ae, oe, ue, and ss, respectively. However, word length was determined based on the standard German notation. 
//------

GENERAL WORD CHARACTERISTICS:
word	Name of the German adjectives 
word_en	English translation of the German adjectives
length	Word length in letters
freq	Frequency per million words in written texts	
freq_cl	Frequency class


WORD MEANS BY SUBGROUPS IN STUDY 1
Valence: 
val_yf	Valence by Young Females
val_ym	Valence by Young Males
val_of	Valence by Old Females
val_om	Valence by Old Males
val_y	Valence by Young adults
val_o	Valence by Old adults
val_t	Valence by Total sample
Arousal:
aro_yf	Arousal by Young Females
aro_ym	Arousal by Young Males
aro_of	Arousal by Old Females
aro_om	Arousal by Old Males
aro_y	Arousal by Young adults
aro_o	Arousal by Old adults
aro_t	Arousal by Total sample
Control:
con_yf	Control by Young Females
con_ym	Control by Young Males
con_of	Control by Old Females
con_om	Control by Old Males
con_y	Control by Young adults
con_o	Control by Old adults
con_t	Control by Total sample
Imagery:
img_yf	Imagery by Young Females
img_ym	Imagery by Young Males
img_of	Imagery by Old Females
img_om	Imagery by Old Males
img_y	Imagery by Young adults
img_o	Imagery by Old adults
img_t	Imagery by Total sample
Self-Relevance: 
ego_yf	Self-Relevance by Young Females
ego_ym	Self-Relevance by Young Males
ego_of	Self-Relevance by Old Females
ego_om	Self-Relevance by Old Males
ego_y	Self-Relevance by Young adults
ego_o	Self-Relevance by Old adults
ego_t	Self-Relevance by Total sample
Age-Relevance: 
age_yf	Age-Relevance by Young Females
age_ym	Age-Relevance by Young Males
age_of	Age-Relevance by Old Females
age_om	Age-Relevance by Old Males
age_y	Age-Relevance by Young adults
age_o	Age-Relevance by Old adults
age_t	Age-Relevance by Total sample


WORD MEANS FOR SELF-OTHER-RELEVANCE IN STUDY 2: 
val_self 	Valence if attributed to the self
val_other	Valence if attributed to others
selfother	Is this a self- or other-relevant attribute? - Mean
selfother_sd	Is this a self- or other-relevant attribute? - SD 
selfother_cl	Classification in self-relevant (2) and other-relevant (1) 


ANALYSES OF VARIANCE ON INDIVIDUAL WORDS IN STUDY 1:
Valence: 
val_age_p	Valence: Age effect – p-values
val_age_e	Valence: Age effect – eta-square
val_age_s	Valence: Age effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
val_sex_p	Valence: Sex effect – p-values
val_sex_e	Valence: Sex effect – eta-square
val_sex_s	Valence: Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
val_axs_p	Valence: Age x Sex effect – p-values
val_axs_e	Valence: Age x Sex effect - eta-square
val_axs_s	Valence: Age x Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
Arousal: 
aro_age_p	Arousal: Age effect – p-values
aro_age_e	Arousal: Age effect – eta-square
aro_age_s	Arousal: Age effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
aro_sex_p	Arousal: Sex effect – p-values
aro_sex_e	Arousal: Sex effect – eta-square
aro_sex_s	Arousal: Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
aro_axs_p	Arousal: Age x Sex effect – p-values
aro_axs_e	Arousal: Age x Sex effect - eta-square
aro_axs_s	Arousal: Age x Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
Control: 
con_age_p	Control: Age effect – p-values
con_age_e	Control: Age effect – eta-square
con_age_s	Control: Age effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
con_sex_p	Control: Sex effect – p-values
con_sex_e	Control: Sex effect – eta-square
con_sex_s	Control: Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
con_axs_p	Control: Age x Sex effect – p-values
con_axs_e	Control: Age x Sex effect - eta-square
con_axs_s	Control: Age x Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
Imagery: 
img_age_p	Imagery: Age effect – p-values
img_age_e	Imagery: Age effect – eta-square
img_age_s	Imagery: Age effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
img_sex_p	Imagery: Sex effect – p-values
img_sex_e	Imagery: Sex effect – eta-square
img_sex_s	Imagery: Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
img_axs_p	Imagery: Age x Sex effect – p-values
img_axs_e	Imagery: Age x Sex effect - eta-square
img_axs_s	Imagery: Age x Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
Self-Relevance: 
ego_age_p	Self-Relevance: Age effect – p-values
ego_age_e	Self-Relevance: Age effect – eta-square
ego_age_s	Self-Relevance: Age effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
ego_sex_p	Self-Relevance: Sex effect – p-values
ego_sex_e	Self-Relevance: Sex effect – eta-square
ego_sex_s	Self-Relevance: Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
ego_axs_p	Self-Relevance: Age x Sex effect – p-values
ego_axs_e	Self-Relevance: Age x Sex effect - eta-square
ego_axs_s	Self-Relevance: Age x Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
Age-Relevance: 
age_age_p	Age-Relevance: Age effect – p-values
age_age_e	Age-Relevance: Age effect – eta-square
age_age_s	Age-Relevance: Age effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
age_sex_p	Age-Relevance: Sex effect – p-values
age_sex_e	Age-Relevance: Sex effect – eta-square
age_sex_s	Age-Relevance: Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)
age_axs_p	Age-Relevance: Age x Sex effect – p-values
age_axs_e	Age-Relevance: Age x Sex effect - eta-square
age_axs_s	Age-Relevance: Age x Sex effect significant?
		0 = not significant (p > .05)
		1 = significant (p < .05)


