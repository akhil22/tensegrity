<mujoco model="dbar"> <include file="./common/visual.xml"/>  <include file="./common/skybox.xml"/> <include file="./common/materials.xml"/>
	
    <!-- C:\Users\Vishala\Documents\MS_THO\PhD_Fall_2019\Study\RL\Project\common -->
    <!--    <option timestep="0.01" iterations="50" solver="Newton" tolerance="1e-10" gravity = "0 0 0" collision="predefined" viscosity="0"/> -->

    <option timestep="0.002" iterations="50" solver="PGS" tolerance="1e-10" gravity = "0 0 0" viscosity="0"/>
    <size njmax="100" nconmax="20" nstack="40000"/>

    <visual>
        <!--<rgba haze=".15 .25 .35 1"/> -->
    </visual>

    <default>
        <!--<joint type="hinge" pos="0 0 0" axis="0 1 0" limited="false" range="-180 180" damping="0"/> -->
        <motor ctrllimited="false" ctrlrange="-1 1"/>
		<tendon stiffness="0.9" damping=".001" range="0.1 0.8"/>
		<geom size="0.02" mass="1"/>
		<site size="0.04"/>
       <camera pos="0 0 10"/>
    </default>
    <asset><texture type="skybox" builtin="gradient" width="100" height="100" rgb1=".4 .6 .8" rgb2="0 0 0"/><texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/><texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="100" height="100"/><material name="MatPlane" reflectance="0.5" texture="texplane" texrepeat="1 1" texuniform="true"/><material name="geom" texture="texgeom" texuniform="true"/></asset>

    <worldbody>
        <geom name="floor" pos="-1 0 0" euler="0 90 0" size="10 10 0.125" type="plane" material="MatPlane" condim="3"/>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
        <site name="s0" pos="0 0 0"/>
	<body>  
      	   <geom name="torso" type="box" pos="1 0 0" size="0.5 0.1 0.4" material="self"/>
	   <site name="At1" pos="-0.3800046 -0.3306 -0.15456"/>
           <site name="At2" pos="0.04222 0.0367 -0.0350"/>
	  
		<body name="body_bar00L"> 		     
	      	     <geom name="bar00L" type="capsule" fromto="0.5 0 -0.4 0.4 0 -0.4" material="self"/>
      	             <geom name="bar0L" type="capsule" fromto="0.4 0 -0.3 0.4 0 -0.5" material="self"/>
	 	     <geom name="bar3L" type="capsule" fromto="-0.1679 -0.00258 -0.04  0.3919 0.00603 -0.04" material="self"/>
	             <site name="b3L" pos="-0.1679 -0.00258 -0.04"/>
	             <site name="b1L" pos="0.3919 0.00603 -0.04"/>	 
     	             <geom name="bar4L" type="capsule" fromto="-0.1679 -0.00258 0.04  0.3919 0.00603 0.04" material="self"/>
	             <site name="b9L" pos="-0.1679 -0.00258 0.04"/>
	             <site name="b7L" pos="0.3919 0.00603 0.04"/>
		     <site name="At3" pos="-0.3800046 -0.3306 -0.15456"/>
	             <site name="At4" pos="0.04222 0.0367 -0.0350"/>
		     <joint name="t2bar00L" type="hinge" pos="0.5 0 -0.4" limited="false" damping="0" armature="0" stiffness="0.2"/> 
		</body>
	</body>
       <!-- <body name="body_bar0L">  
      	 <geom name="bar0L" type="capsule" fromto="0.4 0 -0.3 0.4 0 -0.5" material="self"/>
	 <site name="t2bar0L" pos="-0.3800046 -0.3306 -0.15456"/>
      	 <site name="At8" pos="0.04222 0.0367 -0.0350"/> 
	<joint name="r1" type="free" pos="0 0 0" limited="false" damping="0" armature="0" stiffness="0.2"/>	
        </body> -->
        
        <body>  
         <geom name="bar1L" type="capsule" fromto="-0.3800046 -0.3306 -0.15456 0.04222 0.0367 -0.0350" material="self"/>
	 <site name="b2L" pos="-0.3800046 -0.3306 -0.15456"/>
         <site name="b4L" pos="0.04222 0.0367 -0.0350"/>
	 <joint name="r1L" type="free" pos="0 0 0" limited="false" damping="0" armature="0" stiffness="0.2"/>  
	</body>	
	
        <body>
	  <geom name="bar2L" type="capsule" fromto="-0.3800046 -0.3306 0.15456  0.04222 0.0367 0.0350" material="self"/>
	  <site name="b8L" pos="-0.3800046 -0.3306 0.15456"/>
	  <site name="b10L" pos="0.04222 0.0367 0.0350"/>
	  <joint name="r2L" type="free" pos="0 0 0" limited="false" damping="0" armature="0" stiffness="0.2"/> 
	</body>
	
        <!--<body>
	 <geom name="bar3L" type="capsule" fromto="-0.1679 -0.00258 -0.04  0.3919 0.00603 -0.04" material="self"/>
	 <site name="b3L" pos="-0.1679 -0.00258 -0.04"/>
	 <site name="b1L" pos="0.3919 0.00603 -0.04"/>	 
	 <joint name="r3L" type="free" pos="0 0 1" limited="false" damping="0" armature="0" stiffness="0.2"/> 
    	</body>
	
        <body>
     	 <geom name="bar4L" type="capsule" fromto="-0.1679 -0.00258 0.04  0.3919 0.00603 0.04" material="self"/>
	  <site name="b9L" pos="-0.1679 -0.00258 0.04"/>
	  <site name="b7L" pos="0.3919 0.00603 0.04"/>
-	 <joint name="r4L" type="free" pos="0 0 1" limited="false" damping="0" armature="0" stiffness="0"/> 
    	</body> -->
	<body>
    	<geom name="bar5L" type="capsule" fromto="0.020 -0.05 -0.065  0.020 -0.05 0.065" material="self"/>
	 <site name="b5L" pos="0.020 -0.05 -0.065"/>
	 <site name="b6L" pos="0.020 -0.05 0.065"/>
-	 <joint name="r5L" type="free" pos="0 0 1" limited="false" damping="0" armature="0" stiffness="0"/> 
    	</body> 

		 
	


<!--		<body>  
	      	  <geom name="bar00R" type="capsule" fromto="-0.3800046 -0.3306 -0.15456 0.04222 0.0367 -0.0350" material="self"/>
		  <site name="At5" pos="-0.3800046 -0.3306 -0.15456"/>
	          <site name="At6" pos="0.04222 0.0367 -0.0350"/>
		  <joint name="r1" type="free" pos="0 0 0" limited="false" damping="0" armature="0" stiffness="0.2"/> 


		  <body>  
		      <geom name="bar0R" type="capsule" fromto="-0.3800046 -0.3306 -0.15456 0.04222 0.0367 -0.0350" material="self"/>
		      <site name="At9" pos="-0.3800046 -0.3306 -0.15456"/>
		      <site name="At10" pos="0.04222 0.0367 -0.0350"/>
		      <joint name="r1" type="free" pos="0 0 0" limited="false" damping="0" armature="0" stiffness="0.2"/>

			<body>
			  <geom name="bar2R" type="capsule" fromto="-0.3800046 -0.3306 0.15456  0.04222 0.0367 0.0350" material="self"/>
			  <site name="b8R" pos="-0.3800046 -0.3306 0.15456"/>
			  <site name="b10R" pos="0.04222 0.0367 0.0350"/>
			  <joint name="r2R" type="free" pos="0 0 0" limited="false" damping="0" armature="0" stiffness="0.2"/> 
			</body>
			<body>
			 <geom name="bar3R" type="capsule" fromto="-0.1679 -0.00258 -0.04  0.3919 0.00603 -0.04" material="self"/>
			 <site name="b3R" pos="-0.1679 -0.00258 -0.04"/>
			 <site name="b1R" pos="0.3919 0.00603 -0.04"/>	 
			 <joint name="r3R" type="free" pos="0 0 1" limited="false" damping="0" armature="0" stiffness="0.2"/> 
		    	</body>
			<body>
		     	<geom name="bar4R" type="capsule" fromto="-0.1679 -0.00258 0.04  0.3919 0.00603 0.04" material="self"/>
			 <site name="b9R" pos="-0.1679 -0.00258 0.04"/>
			 <site name="b7R" pos="0.3919 0.00603 0.04"/>
		-	 <joint name="r4R" type="free" pos="0 0 1" limited="false" damping="0" armature="0" stiffness="0"/> 
		    	</body>
			<body>
		    	<geom name="bar5R" type="capsule" fromto="0.020 -0.05 -0.065  0.020 -0.05 0.065" material="self"/>
			 <site name="b5R" pos="0.020 -0.05 -0.065"/>
			 <site name="b6R" pos="0.020 -0.05 0.065"/>
		-	 <joint name="r5R" type="free" pos="0 0 1" limited="false" damping="0" armature="0" stiffness="0"/> 
		    	</body> 
		</body>
  	    </body>
	  
	  
	  
	</body> -->


       <!--  <body>
	 <geom name="floor" pos="0 0 -0.5" size="0 0 0.5" type="plane" material="grid"/>
	 <site name="xs4" pos="-2 0 0"/>
	 <site name="xs5" pos="2 0 0"/>
	 <site name="target" pos="0 0 2.5" size="0.08" rgba="1 0 0 .3"/>
       </body> -->
    </worldbody>	
<!--<equality>
		<connect active="true" name='1bar34' body1="body_bar00L" body2="body_bar0L" anchor="0.4 0 -0.4"/>
</equality>  -->
  <tendon>
        <spatial name="S1L" width="0.02">
            <site site="b2L"/>
            <site site="b3L"/>
        </spatial>
  	<spatial name="S2L" width="0.02">
            <site site="b3L"/>
            <site site="b4L"/>
        </spatial>
	<spatial name="S3L" width="0.02">
            <site site="b4L"/>
            <site site="b1L"/>
        </spatial>
	<spatial name="S4L" width="0.02">
            <site site="b1L"/>
            <site site="b5L"/>
        </spatial>
	<spatial name="S5L" width="0.02">
            <site site="b2L"/>
            <site site="b5L"/>
        </spatial>
	<spatial name="S6L" width="0.02">
            <site site="b3L"/>
            <site site="b5L"/>
        </spatial>
	<spatial name="S7L" width="0.02">
            <site site="b4L"/>
            <site site="b5L"/>
        </spatial>
	<spatial name="S8L" width="0.02">
            <site site="b1L"/>
            <site site="b6L"/>
        </spatial>
	<spatial name="S9L" width="0.02">
            <site site="b2L"/>
            <site site="b6L"/>
        </spatial>
	<spatial name="S10L" width="0.02">
            <site site="b3L"/>
            <site site="b6L"/>
        </spatial>
	<spatial name="S11L" width="0.02">
            <site site="b4L"/>
            <site site="b6L"/>
        </spatial>
	<spatial name="S12L" width="0.02">
            <site site="b8L"/>
            <site site="b9L"/>
        </spatial>
	<spatial name="S13L" width="0.02">
            <site site="b9L"/>
            <site site="b10L"/>
        </spatial>
	<spatial name="S14L" width="0.02">
            <site site="b10L"/>
            <site site="b7L"/>
        </spatial>
	<spatial name="S15L" width="0.02">
            <site site="b5L"/>
            <site site="b7L"/>
        </spatial>
	<spatial name="S16L" width="0.02">
            <site site="b5L"/>
            <site site="b8L"/>
        </spatial>
	<spatial name="S17L" width="0.02">
            <site site="b5L"/>
            <site site="b9L"/>
        </spatial>
      <spatial name="S18L" width="0.02">
            <site site="b5L"/>
            <site site="b10L"/>
	</spatial>
	<spatial name="S19L" width="0.02">
            <site site="b6L"/>
            <site site="b7L"/>
        </spatial>
	<spatial name="S20L" width="0.02">
            <site site="b6L"/>
            <site site="b8L"/>
        </spatial>
	<spatial name="S21L" width="0.02">
            <site site="b6L"/>
            <site site="b9L"/>
        </spatial>
	<spatial name="S22L" width="0.02">
            <site site="b6L"/>
            <site site="b10L"/>
        </spatial>
	<spatial name="S23L" width="0.02">
            <site site="b4L"/>
            <site site="b10L"/>
        </spatial>

 <!--       <spatial name="S1R" width="0.02">
            <site site="b2R"/>
            <site site="b3R"/>
        </spatial>
  	<spatial name="S2R" width="0.02">
            <site site="b3R"/>
            <site site="b4R"/>
        </spatial>
	<spatial name="S3R" width="0.02">
            <site site="b4R"/>
            <site site="b1R"/>
        </spatial>
	<spatial name="S4R" width="0.02">
            <site site="b1R"/>
            <site site="b5R"/>
        </spatial>
	<spatial name="S5R" width="0.02">
            <site site="b2R"/>
            <site site="b5R"/>
        </spatial>
	<spatial name="S6R" width="0.02">
            <site site="b3R"/>
            <site site="b5R"/>
        </spatial>
	<spatial name="S7R" width="0.02">
            <site site="b4R"/>
            <site site="b5R"/>
        </spatial>
	<spatial name="S8R" width="0.02">
            <site site="b1R"/>
            <site site="b6R"/>
        </spatial>
	<spatial name="S9R" width="0.02">
            <site site="b2R"/>
            <site site="b6R"/>
        </spatial>
	<spatial name="S10R" width="0.02">
            <site site="b3R"/>
            <site site="b6R"/>
        </spatial>
	<spatial name="S11R" width="0.02">
            <site site="b4R"/>
            <site site="b6R"/>
        </spatial>
	<spatial name="S12R" width="0.02">
            <site site="b8R"/>
            <site site="b9R"/>
        </spatial>
	<spatial name="S13R" width="0.02">
            <site site="b9R"/>
            <site site="b10R"/>
        </spatial>
	<spatial name="S14R" width="0.02">
            <site site="b10R"/>
            <site site="b7R"/>
        </spatial>
	<spatial name="S15R" width="0.02">
            <site site="b5R"/>
            <site site="b7R"/>
        </spatial>
	<spatial name="S16R" width="0.02">
            <site site="b5R"/>
            <site site="b8R"/>
        </spatial>
	<spatial name="S17R" width="0.02">
            <site site="b5R"/>
            <site site="b9R"/>
        </spatial>
      <spatial name="S18R" width="0.02">
            <site site="b5R"/>
            <site site="b10R"/>
	</spatial>
	<spatial name="S19R" width="0.02">
            <site site="b6R"/>
            <site site="b7R"/>
        </spatial>
	<spatial name="S20R" width="0.02">
            <site site="b6R"/>
            <site site="b8R"/>
        </spatial>
	<spatial name="S21R" width="0.02">
            <site site="b6R"/>
            <site site="b9R"/>
        </spatial>
	<spatial name="S22R" width="0.02">
            <site site="b6R"/>
            <site site="b10R"/>
        </spatial>
	<spatial name="S23R" width="0.02">
            <site site="b4R"/>
            <site site="b10R"/>
        </spatial> -->
    </tendon> 

   <actuator>
        <motor tendon="S1L" gear="1"/>
        <motor tendon="S2L" gear="1"/>
		<motor tendon="S3L" gear="1"/>
        <motor tendon="S4L" gear="1"/>
		<motor tendon="S5L" gear="1"/>
        <motor tendon="S6L" gear="1"/>
		<motor tendon="S7L" gear="1"/>
        <motor tendon="S8L" gear="1"/>
		<motor tendon="S9L" gear="1"/>
        <motor tendon="S10L" gear="1"/>
		<motor tendon="S11L" gear="1"/>
        <motor tendon="S12L" gear="1"/>
	    <motor tendon="S13L" gear="1"/>
        <motor tendon="S14L" gear="1"/>
		<motor tendon="S15L" gear="1"/>
        <motor tendon="S16L" gear="1"/>
		<motor tendon="S17L" gear="1"/>
		<motor tendon="S18L" gear="1"/>
        <motor tendon="S19L" gear="1"/>
		<motor tendon="S20L" gear="1"/>
        <motor tendon="S21L" gear="1"/>
	    <motor tendon="S22L" gear="1"/>
        <motor tendon="S23L" gear="1"/>
  <!--      <motor tendon="S1R" gear="1"/>
        <motor tendon="S2R" gear="1"/>
		<motor tendon="S3R" gear="1"/>
        <motor tendon="S4R" gear="1"/>
		<motor tendon="S5R" gear="1"/>
        <motor tendon="S6R" gear="1"/>
		<motor tendon="S7R" gear="1"/>
        <motor tendon="S8R" gear="1"/>
		<motor tendon="S9R" gear="1"/>
        <motor tendon="S10R" gear="1"/>
		<motor tendon="S11R" gear="1"/>
        <motor tendon="S12R" gear="1"/>
	    <motor tendon="S13R" gear="1"/>
        <motor tendon="S14R" gear="1"/>
		<motor tendon="S15R" gear="1"/>
        <motor tendon="S16R" gear="1"/>
		<motor tendon="S17R" gear="1"/>
		<motor tendon="S18R" gear="1"/>
        <motor tendon="S19R" gear="1"/>
		<motor tendon="S20R" gear="1"/>
        <motor tendon="S21R" gear="1"/>
	    <motor tendon="S22R" gear="1"/>
        <motor tendon="S23R" gear="1"/> -->
   </actuator> 
	
	<sensor>
		<framelinvel objtype="site" objname="b1L"/>
		<framelinvel objtype="site" objname="b2L"/>
		<framelinvel objtype="site" objname="b3L"/>
		<framelinvel objtype="site" objname="b4L"/>
		<framelinvel objtype="site" objname="b5L"/>
		<framelinvel objtype="site" objname="b6L"/>
        	<framelinvel objtype="site" objname="b7L"/>
		<framelinvel objtype="site" objname="b8L"/>
		<framelinvel objtype="site" objname="b9L"/>
		<framelinvel objtype="site" objname="b10L"/>
<!--		<framelinvel objtype="site" objname="b1R"/>
		<framelinvel objtype="site" objname="b2R"/>
		<framelinvel objtype="site" objname="b3R"/>
	<framelinvel objtype="site" objname="b4R"/>
		<framelinvel objtype="site" objname="b5R"/>
		<framelinvel objtype="site" objname="b6R"/>
        	<framelinvel objtype="site" objname="b7R"/>
		<framelinvel objtype="site" objname="b8R"/>
		<framelinvel objtype="site" objname="b9R"/>
		<framelinvel objtype="site" objname="b10R"/> -->
	</sensor>
</mujoco>
