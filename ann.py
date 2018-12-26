class ann:
	
	def __init__(this, layers = [], name = ''):	
		this.name = name
		this.layers = layers		
		
	def __str__(this):
		return this.__class__.__name__ + " class with name " + this.name
		
		
class ann_by_layers(ann):
	
	def __init__(this, layers = [], name = ''):	
		this.name = name
		this.layers = layers		
		
	def __str__(this):
		return this.__class__.__name__ + " class with name " + this.name
				