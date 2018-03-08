# CV


This is a subproject of BOThello to quickly label games into datasets. To do that, the main two files are
[CV_Gui.py](CV_Gui.py) and [CV_Process.py](CV_Process.py).

[CV_Gui.py](CV_Gui.py) is, as the name suggests, a GUI-wrapper of the CV-project, which is the main way to label games.
It generates a file containing the board state for each move in the game, commonly called just `boards`.

[CV_Process.py](CV_Process.py) is much simpler, it converts the board state list into the dataset format used by the AI,
which is just a list of the moves made.

## Screenshot

<img src="https://github.com/loovjo/BOThello/blob/master/CV/screenshot.png" width=700px/>
