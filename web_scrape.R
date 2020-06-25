#pkg 
library(data.table)
library(readxl)
library(stringr)
library(rvest)
library(downloader)
library(tools)

#import data

lgs <- data.table(read_xlsx("data_cross_comparison.xlsx", sheet = "token counts"))
tbl <- data.table(read_xlsx("data_cross_comparison.xlsx", sheet = "table of doom"))

#split strings and unlist - take out empty spaces, NA's and "Modern(1453–)" from Greek
#reassign to lgs
lgs <- na.omit(trimws(unlist(str_split(lgs$Language_long, ",|;"))))[
  na.omit(gsub(" ","",unlist(str_split(lgs$Language_long, ",|;")))) != "Modern(1453–)"
]

#take Portuguese out of 45 
lgs[45] <- "Brazilian"


ref <- data.table()

#loop
for(i in 1:length(lgs)){
  ref <- rbind(ref, tbl[language == lgs[i] & aoa == 1 | 
        language == lgs[i] & concrete == 1 | 
        language == lgs[i] & familiar == 1 | 
        language == lgs[i] & imagine == 1 | 
        language == lgs[i] & valence == 1, .(language,author,bibtex,ref_title, ref_doi,aoa,concrete,familiar,imagine,valence)]
  )
}

#create unified "variable" variable
ref[, aoa := ifelse(ref$aoa == 1, "AOA", "")][]
ref[, concrete := ifelse(ref$concrete == 1, "concrete", "")][]
ref[, familiar := ifelse(ref$familiar == 1, "familiar", "")][]
ref[, imagine := ifelse(ref$imagine == 1, "imagine", "")][]
ref[, valence := ifelse(ref$valence == 1, "valence", "")][]

ref[, variable := trimws(paste(ref$aoa, ref$concrete, ref$familiar, ref$imagine, ref$valence))][]

#sort decreasing so I can start from last language
setorder(ref, -language)

#use ref titles to find links that we can search for "electronic supplementary material"

# #start with google scholar link --- uhhhh never mind Google Scholar does NOT appreciate my web scraping skills
# base_url <- "http://scholar.google.com/scholar?"

# create data.frame
log <- data.frame()

# set up base url
base_url <- "https://link.springer.com/article/"

for(i in 1:nrow(ref)){
  tryCatch({ 
  print(paste("starting with line", i))
  #add in our "search" and create a url for google scholar
  print(i)
  if(is.na(ref[i, ref_doi])){next}
  (search_string <- ref[i, ref_doi]) ########
  url <- paste0(base_url,search_string)
  
  #capture html 
  print("getting page url")
  links <- url
  
  #get unique vector of links I'm actually interested in and pulling the article link
  print("getting vector of links")
  article_url <- grep("article", unique(grep("springer", links, value = T)), value = T)
  
  #read in article URL
  print("read article url")
  article_pg <- read_html(article_url)
  
  #get links to supplemental materials
  print("links to supplemental")
  html_attr(html_nodes(article_pg, '.print-link'), 'href')
  
  file_names <- rep(as.character(NA), length(html_attr(html_nodes(article_pg, '.print-link'), 'href')))
  
  print(i)
  for(num in 1:length(html_attr(html_nodes(article_pg, '.print-link'), 'href'))){
    print(num)
    file_names[num] <- paste0(ref[i,]$author, "_", num, ".", file_ext(html_attr(html_nodes(article_pg, '.print-link'), 'href')))[num]
  }
  
  #log the entries
  print("logging entries")
  if(length(html_attr(html_nodes(article_pg, '.print-link'), 'href')) == 0){next}
  log <- rbind(log, data.frame(language = ref[i,]$language,
                                        variable = ref[i,]$variable,
                                        file_name = file_names
  ))
  
  #save the files
  for(j in seq_along(html_attr(html_nodes(article_pg, '.print-link'), 'href'))){
    tryCatch(download(url = html_attr(html_nodes(article_pg, '.print-link'), 'href')[j], paste0("datasets/low_hanging/", file_names[j])))
  }
  
  #brief nap
  Sys.sleep(1)
  }, error = function(e){cat("Error: ", conditionMessage(e), "\n")})
}

#check log
log

#write out log
fwrite(log, "datasets/low_hanging/datasets_found_gc.csv", quote = F, sep = ",")
