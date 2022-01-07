@REM *** INSTRUCTIONS ***
@REM 1) Copy and rename this batch file to the same name as the .py python script you want to run
@REM    ex: 'Report1.py' --> 'Report1.bat' etc
@REM 2) Replace <path to anaconda3> with the correct path on your system
@REM 3) Make sure this batch file is in the same folder as the .py script
@REM 4) Run this batch file to run the .py script. You can also make a shortcut to this
       batch file and run it from the shortcut (ex: from your desktop etc)

call C:\Users\mdaws\anaconda3\Scripts\activate.bat
call %~n0.py
@cmd /k