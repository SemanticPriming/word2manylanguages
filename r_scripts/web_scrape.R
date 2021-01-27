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
fwrite(log, "datasets/low_hanging/datasets_found_gc.csv", quote = T, sep = ",")

#start to clean up log
log <- data.table(log)

#get rid of all pdf, gif, docs, and zip
log2 <- log[!file_name %in% grep(c(".pdf|.gif|.doc|.zip"), log$file_name, value = T),]

#place all zip files at the bottom 
log2 <- rbind(log2,log[file_name %in% grep(c(".zip"), log$file_name, value = T),])

#create file search variable to help with grepping files later
f_search <- character()

#split and grab first word from files/authors
for(i in 1:length(str_split(log2$file_name, boundary("word")))){
  f_search[i] <- str_split(log2$file_name, boundary("word"))[[i]][1]
}

#loop for datasets that are just one file
for(i in 74:nrow(log2)){
  print(i)
  if(length(grep(f_search[i], list.files("datasets/low_hanging/data_files/"), value = T, ignore.case = T)) > 1){next}
  tryCatch({
    log2[file_name %in% grep(f_search[i], log2$file_name, value = T),
         file_name := grep(f_search[i], list.files("datasets/low_hanging/data_files/", ignore.case = T), value = T)][]
  }, error = function(e){cat("Error: ", conditionMessage(e), "\n")})
}

#add file search variable to data.table
log2[, f_search := f_search]

#make a new log dt without any of the zip files
log3 <- log2[!grep(".zip", log2$file_name)]

#row bind to include all multiple file datasets 
for(i in 1:nrow(log2[grep(".zip", log2$file_name)])){
  log3 <- rbind(log3, 
                data.table(language = log2[grep(".zip", log2$file_name)][i]$language,
                           variable = log2[grep(".zip", log2$file_name)][i]$variable, 
                           file_name = grep(log2[grep(".zip", log2$file_name)][i]$f_search, 
                                            list.files("datasets/low_hanging/data_files/"), 
                                            ignore.case = T, value = T)),
                fill = T)
}

#delete file search var
log3[, f_search := NULL][]

#sort by language
setorder(log3, -language)

#and write out. 
fwrite(log3, "datasets/low_hanging/datasets_found_gc.csv", quote = T, sep = ",")
