## Satellite_analysis

#satellite_image_script se usara para analisis puntuales y pruebas. aqui se puede definir usuario/terreno, fechas de analisis y si se descarga o no imagenes.
	- este va a ser eliminado proximamente, ya que el "_v3" puede hacer lo mismo, al definir usuario o fechas como input
#satellite_image_script_v3:
	- para ejecutar en CMD = python satellite_image_script_v3.py --download yes (automatizado para descargar mas reciente solicitud de firebase con fechas de usuario)
	- hay varios user inputs que se pueden agregar despues del script, para hacer cosas diferentes:
		* --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-MAa0O5PMyE81I_AFC6E, analiza el usuario/terreno definido (se puede cambiar el usuario/terreno a uno propio)
		* --date_ini 2020-01-01 --date_fin 2020-01-31, cambia las fechas definidas en firebase, por las aqui ingresadas
		* --download no, o si se omite --download, no descargara imagenes de la API, y buscara archivos en el PC, dentro de la carpeta 'Unzipped_Images'
	- Descarga automática del código de usuario y código terreno para la solicitud más reciente
	- Correccción de los nombres con los que se guardan lás graficas de estadísticas
	- Simplificación del código fire_up, el cual también modifica la variable "Status" de firebase al estado "Finalizado", 
	  para saber las solicitudes que ya han sido entregadas
