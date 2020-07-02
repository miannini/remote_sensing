## Satellite_analysis

satellite_image_script se usara para analisis puntuales y pruebas. aqui se puede definir usuario/terreno, fechas de analisis y si se descarga o no imagenes.

satellite_image_script_v3:
	- Descarga automática del código de usuario y código terreno para la solicitud más reciente
	- Correccción de los nombres con los que se guardan lás graficas de estadísticas
	- Simplificación del código fire_up, el cual también modifica la variable "Status" de firebase al estado "Finalizado", 
	  para saber las solicitudes que ya han sido entregadas