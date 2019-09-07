@echo off
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_1_ratio_0.1.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_1_ratio_0.1.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_1_ratio_0.4.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_1_ratio_0.4.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt

set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_1_ratio_0.7.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_1_ratio_0.7.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_1_ratio_0.9.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_1_ratio_0.9.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt


@echo off
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_2_ratio_0.1.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_2_ratio_0.1.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_2_ratio_0.4.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_2_ratio_0.4.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt

set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_2_ratio_0.7.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_2_ratio_0.7.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_2_ratio_0.9.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_2_ratio_0.9.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt



@echo off
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_4_ratio_0.1.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_4_ratio_0.1.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_4_ratio_0.4.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_4_ratio_0.4.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt

set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_4_ratio_0.7.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_4_ratio_0.7.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_4_ratio_0.9.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_4_ratio_0.9.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt



@echo off
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_6_ratio_0.1.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_6_ratio_0.1.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_6_ratio_0.4.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_6_ratio_0.4.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt

set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_6_ratio_0.7.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_6_ratio_0.7.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_6_ratio_0.9.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_6_ratio_0.9.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt



@echo off
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_8_ratio_0.1.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_8_ratio_0.1.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_8_ratio_0.4.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_8_ratio_0.4.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt

set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_8_ratio_0.7.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_8_ratio_0.7.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt
set "xprvar1="
for /F "skip=36 delims=" %%i in (layers_8_ratio_0.9.txt) do if not defined xprvar1 set "xprvar1=%%i"
for /F "tokens=5" %%i in ("%xprvar1%") do set acc1=%%i
for /F "tokens=6" %%i in ("%xprvar1%") do set acc2=%%i
set "xprvar2="
for /F "skip=53 delims=" %%i in (layers_8_ratio_0.9.txt) do if not defined xprvar2 set "xprvar2=%%i"
for /F "tokens=5" %%i in ("%xprvar2%") do set precision=%%i
for /F "tokens=6" %%i in ("%xprvar2%") do set recall=%%i
for /F "tokens=7" %%i in ("%xprvar2%") do set f1=%%i
Echo %acc1%%acc2% %precision% %recall% %f1%>>output.txt