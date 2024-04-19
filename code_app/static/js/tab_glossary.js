// function to insert content for the "glossary" tab

// define the function
function insertHTML() {

    let newHTML = `
        
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Basin (Geological Basin)</ins>:</b></span><span style="display: block;">A large-scale geological depression, often circular or elliptical in shape, where layers of sediment accumulate over time. Basins are critical in petroleum geology as they are often the sites of significant accumulations of oil and natural gas. These structures are formed by tectonic actions such as subsidence of the Earth's crust and can vary widely in size and complexity.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Charge Balance</ins>:</b></span><span style="display: block;">In geochemistry, this refers to the state where the sum of the charges from all the cations (positively charged ions) and anions (negatively charged ions) in a solution are balanced. It's important for understanding the chemical stability of mineral waters and produced waters.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Class II Injection Well</ins>:</b></span><br><span style="display: block;">A type of well designated for the injection of fluids associated with oil and natural gas production back into the ground. These wells are regulated under the Safe Drinking Water Act and are used to enhance oil production, dispose of brine (saltwater) that is a byproduct of oil and gas production, and store hydrocarbons that are liquid at standard temperature and pressure. Class II wells help in managing the byproducts of oil and gas extraction, thereby mitigating potential environmental impacts.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>CMG (Computer Modelling Group) Model</ins>:</b><br></span><span style="display: block;">A sophisticated computational tool used for simulating subsurface flow and transport phenomena, including hydrocarbon extraction, CO2 sequestration, and groundwater movement. CMG models are essential in the energy industry for reservoir simulation, helping in decision-making for exploration, development, and management of oil and gas resources.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Concentration</ins>:</b></span><span style="display: block;">The abundance of a constituent divided by the total volume of a mixture. In geochemistry, concentration is a fundamental concept used to quantify the level of a particular element or compound in a geological sample. It is critical for understanding the composition and quality of groundwater, surface water, and produced water.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Delaware Basin</ins>:</b></span><br><span style="display: block;">A sub-basin of the Permian Basin located in West Texas and southeastern New Mexico, known for its significant oil and gas production. The Delaware Basin has been a focus of study for induced seismicity related to petroleum extraction activities.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Formation (Geological Formation in Oil and Gas Reservoirs)</ins>:</b></span><span style="display: block;">In the context of oil and gas exploration and production, a geological formation is a distinct layer of sedimentary rock with consistent characteristics that distinguish it from adjacent strata. These formations are critical in identifying potential reservoirs of hydrocarbons. They often contain organic material that, over geological time, has been transformed into oil and gas. The properties of a formation, such as porosity, permeability, and thickness, are key factors in determining the viability and productivity of an oil or gas reservoir. In oil and gas terminology, formations are usually named after the geographic location where they were first studied or identified. Understanding the geological formations is essential for successful drilling and extraction operations, as it guides the placement of wells and informs predictions about the presence and recoverability of oil and gas deposits.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Gridded Numerical Models</ins>:</b></span><br><span style="display: block;">Computational models that represent the subsurface through a grid of cells, allowing for the simulation of processes such as fluid flow and pressure changes. These models are used to predict how changes in conditions might affect the subsurface, including the potential for induced seismicity.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Induced Seismicity</ins>:</b></span><br><span style="display: block;">Earthquakes that result from human activities, such as the injection or extraction of fluids from the earth's subsurface, mining, reservoir-induced seismicity from the filling of large reservoirs, and other large-scale engineering projects.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Injection Volume</ins>:</b></span><br><span style="display: block;">The total amount of fluid injected into a well over a specified period. This term is particularly relevant in the context of SWD wells, where the volume of injected wastewater is a critical factor in understanding the potential for induced seismicity.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Interactive Geomap Analysis Tool</ins>:</b></span><br><span style="display: block;">A digital tool that allows users to visualize and interact with geographic data on a map. In the context of this study, it refers to a tool developed to display seismic data, SWD well injection volumes, and pore pressure changes to analyze their relationships.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Model Layers</ins>:</b></span><span style="display: block;">Refers to the discrete stratigraphic units within a geological basin as represented in the CMG model. Each layer is a simplification of a geological formation or group of formations, characterized by specific properties such as porosity, permeability, and fluid saturation.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Model Layer 9-13 (Devonian-Silurian Top and Bottom Formation)</ins>:</b></span><span style="display: block;">These layers correspond to the geological formations spanning the Devonian and Silurian periods within the model. They are critical for understanding the sedimentary deposits and hydrocarbon potential in the Delaware Basin.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Model Layer 19 (Ellenberger Formation)</ins>:</b></span><span style="display: block;">This layer specifically represents the Ellenberger Formation within the CMG model, a significant carbonate rock formation dating back to the Early Ordovician period.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Major Elements</ins>:</b></span><span style="display: block;">These are the elements found in high concentrations in geological samples. They are significant in geochemical investigations as they influence the chemical and physical properties of rocks and fluids.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Molarity</ins>:</b></span><span style="display: block;">A unit of concentration in chemistry, representing the number of moles of a solute dissolved in one liter of solution. It is a standard measure for quantifying the concentration of elements or compounds in a solution, crucial in geochemical analyses to determine the precise chemical makeup of water samples, including produced water.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Piper Plot</ins>:</b></span><span style="display: block;">A graphical representation used in hydrochemistry to illustrate the chemical composition of water samples. The plot is divided into three fields: two triangular fields that show the major cations (calcium, magnesium, sodium, and potassium) and anions (carbonate, bicarbonate, sulfate, and chloride) respectively, and a central diamond-shaped field that provides a comprehensive view of water chemistry. Piper plots are instrumental in understanding the geochemical evolution of water, identifying water types, and assessing water-rock interaction processes.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Pore Pressure</ins>:</b></span><br><span style="display: block;">The pressure of fluids within the pores of a rock or soil, which can affect the rock's mechanical properties and its ability to transmit fluids. Changes in pore pressure can influence the stability of the rock and potentially trigger seismic events.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Produced Water</ins>:</b></span><span style="display: block;">This refers to the water that is brought to the surface during oil and gas extraction. It often contains various organic and inorganic substances and is typically considered a byproduct of the hydrocarbon extraction process.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Salt Water Disposal (SWD) Well</ins>:</b></span><br><span style="display: block;">A type of well used for the disposal of saline water (brine) that is produced along with oil and gas. The water is injected into porous rock formations deep underground, often into the same formation from which it was produced.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Scaling Elements in Produced Water Treatment</ins>:</b></span><span style="display: block;">In the context of produced water treatment, scaling elements refer to minerals like calcium, magnesium, barium, and strontium, which can precipitate out of produced water under certain conditions. These precipitates can form scale that coats and clogs pipes and equipment, causing significant operational challenges in treatment processes. Managing scaling elements is crucial for efficient and cost-effective treatment of produced water.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Seismic Activity</ins>:</b></span><br><span style="display: block;">The frequency, type, and size of earthquakes experienced over a period of time in a specific area. Seismic activity is a natural process but can be influenced by human activities, such as the injection of fluids into the earth's subsurface.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>TexNet</ins>:</b></span><br><span style="display: block;">The Texas Seismological Network, a state-funded initiative to monitor and research earthquake activity across Texas. TexNet provides public access to seismic data through its catalog.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Total Dissolved Solid (TDS)</ins>:</b></span><span style="display: block;">A measure of the combined content of all inorganic and organic substances contained in a liquid. In water quality analysis and geochemistry, TDS is used to indicate the general quality of the water.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Violin Plot</ins>:</b></span><span style="display: block;">A method of data visualization that combines a box plot with a kernel density plot. In environmental and geochemical studies, violin plots can illustrate the distribution and probability density of data, particularly useful for comparing multiple data sets.<br><br></span>
    </div>
    <div class="row">
        <span class="larger" style="display: block;"><b><ins>Well Depth</ins>:</b><br><br></span><span style="display: block;">The vertical distance measured from the surface to the bottom of a well. This is a critical factor in geological and hydrological studies as it can influence the characteristics of the water or oil extracted.<br><br></span>
    </div>

    `;

    // Insert the new HTML content into the page
    document.getElementById("tab_glossary").innerHTML = newHTML;
}

// call the function
insertHTML();