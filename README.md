# Curso Visión Computacional Aplicada a la Robótica, Maestría en Ingeniería, UNAM

Material para el curso de Visión Computacional Aplicada a la Robótica de la Maestría en Ingeniería Eléctrica, UNAM, Semestre 2023-2

## Requerimientos

* Ubuntu 20.04
* ROS Noetic http://wiki.ros.org/noetic/Installation/Ubuntu
* Webots 2022a: https://github.com/cyberbotics/webots/releases/download/R2022a/webots_2022a_amd64.deb

## Instalación

Nota: se asume que ya se tiene instalado Ubuntu, Webots y ROS.

* $ cd
* $ git clone https://github.com/mnegretev/ComputerVisionForRobotics-2023-2
* $ cd ComputerVisionForRobotics-2023-2
* $ ./Setup.sh
* $ cd catkin_ws
* $ catkin_make -j2 -l2

## Pruebas

Este respositorio contiene simulaciones para un robot de servicio doméstico y para un vehículo sin conductor. Para correr la simulación del robot de servicio, ejecute lo siguiente:

* $ cd 
* $ source ComputerVisionForRobotics-2023-2/catkin_ws/devel/setup.bash
* $ roslaunch surge_et_ambula justina_gazebo.launch

Si todo se instaló y compiló correctamente, se debería ver un visualizador como el siguiente:

<img src="https://github.com/mnegretev/ComputerVisionForRobotics-2023-2/blob/master/Media/rviz.png" alt="RViz" width="720"/>

Un ambiente simulado como el siguiente:

<img src="https://github.com/mnegretev/ComputerVisionForRobotics-2023-2/blob/master/Media/gazebo.png" alt="Gazebo" width="764"/>

Y una GUI como la siguiente:

<img src="https://github.com/mnegretev/ComputerVisionForRobotics-2023-2/blob/master/Media/gui.png" alt="GUI" width="339"/>

Para correr la simulación del vehículo sin conductor, ejecute lo siguiente:

* $ cd 
* $ source ComputerVisionForRobotics-2023-2/catkin_ws/devel/setup.bash
* $ roslaunch webots_simul navigation_no_obstacles.launch

Si todo se instaló y compiló correctamente, se debería ver un ambiente simulado como el siguiente:

<img src="https://github.com/mnegretev/ComputerVisionForRobotics-2023-2/blob/master/Media/webots.png" alt="Webots" width="764"/>

Y una GUI como la siguiente:

<img src="https://github.com/mnegretev/ComputerVisionForRobotics-2023-2/blob/master/Media/webots_gui.png" alt="GUI" width="375"/>


## Máquina virtual

Se puede descargar una máquina virtual para [VirtualBox](https://www.virtualbox.org/wiki/Downloads) con Ubuntu, ROS y Webots ya instalados de [esta dirección.](https://drive.google.com/drive/folders/185jFqv_20o3U70ESKVnhUv1zi7AG2F7u?usp=share_link) <br>
Solo descomprima el archivo y siga las instrucciones del video que se encuentra en la misma carpeta. La máquina virtual ya tiene todo instalado por lo que se puede pasar a la sección de pruebas. 

Se recomienda configurar la máquina virtual con 4 CPUs y 4GB de RAM.<br>
Usuario: cv <br>
Contraseña: pumas


## Contacto
Dr. Marco Negrete<br>
Profesor Asociado C<br>
Departamento de Procesamiento de Señales<br>
Facultad de Ingeniería, UNAM <br>
marco.negrete@ingenieria.unam.edu<br>
