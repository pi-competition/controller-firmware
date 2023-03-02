
## RULES FOR THE ROADMAP:
- intersections in yellow, road dots in white, everything else can be whatever, but no other yellow or white.  
- roads cant just end, cap them off with an intersection  
- intersections have to be a polygon with as many sides as there are roads going into them, unless there's only one road,
in which case the shape doesn't matter. knock yourself out.
- for the above reason you can't have a two-road intersection. use a corner.
  
everything else should be okay i think. try it and see.  
this bad code brought to you by Abbas. direct all complaints to your nearest wall.  

### how do i run this?
```pip install --user pipenv```

```cd controller-firmware && pipenv install``` (pipenv is in `~/.local/bin/pipenv`)

```pipenv run python src/main.py <path_to_ur_image>```
  