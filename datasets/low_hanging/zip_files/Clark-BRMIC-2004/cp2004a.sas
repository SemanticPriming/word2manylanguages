*SAS Commands to read DBF file and select items
*  according to normative values;

filename file1 'cp2004a.dbf' ;
proc dbf db5 = file1;

ods html file='cp2004a.html';

* use * instead of variable names for all variables;
proc sql;
  select word, pls, len from work.data1
  where pls < 2.00 and len <10
  order by pls;

ods html close;

