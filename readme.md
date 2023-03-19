
## RULES FOR THE ROADMAP:
- intersections in yellow, road dots in white, everything else can be whatever, but no other yellow or white.  
- roads cant just end, cap them off with an intersection  
- intersections have to be a polygon with as many sides as there are roads going into them, unless there's only one road,
in which case the shape doesn't matter. knock yourself out.
- for the above reason you can't have a two-road intersection. use a corner.
  
everything else should be okay i think. try it and see.  
this bad code brought to you by Abbas. direct all complaints to your nearest wall.  

### how do i run this?
sort out the venv or whatever idk how it works
source something in the bin
  
then `python3 src/main.py`  
wait a bit, it'll prompt for press enter for to take a map image then present the image.  
if you're satisfied close the popup and reply y, else reply n.  
It then does some maths magic and shows some more (mostly diagnostic but pretty cool) popups, which should be closed.
Then it'll just do car stuff.
