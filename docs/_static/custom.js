(function (){

  $(function(){
    $("dl.py.method dt:has(em.property)").addClass("dt-property");
  });

  $(function(){

    var items = document.querySelectorAll('div.sphinxsidebarwrapper li[class*="toctree-"]');
    items.forEach((item) => {
      var text  = item.querySelector("a").innerHTML
      var text_components = text.split(".");
      var num_components = text_components.length;
      if (num_components>0){
        text = text_components[num_components-1];
      }
      text = text.replace(/\spackage/g,' <span class="package-name">package</span>')
                .replace(/(.*)\smodule/g, '<span class="program-name">$1.py</span>')
                .replace(/(Subpackages|Submodules)/g,'<span class="package-subtitle">$1</span>');
      item.querySelector("a").innerHTML = text
    });

  });
})(jQuery);

window.onload = function() {
  Particles.init({
    selector: '.particle-background',
    color: ["#cee6b4", "#1f441e"],
    maxParticles: 300,
    connectParticles: true,
    sizeVariations: 5,
    responsive: [
      {
        breakpoint: 1100,
        options: {
          maxParticles: 0, // disables particles.js for speed.
        }
      }
    ]
    
  });
  Particles._resize()
};

// Utility Functions
const changeParticles = function(e){
  if (Particles.options.maxParticles==0){
    Particles.options.maxParticles = 300;
    e.textContent = "Stop Particles";
    e.classList.add("btn-stop");
    e.classList.remove("btn-start");
  }else{
    Particles.options.maxParticles = 0;
    e.textContent = "Start Particles";
    e.classList.add("btn-start");
    e.classList.remove("btn-stop");
  }
  Particles._refresh();
  return false;
}

const changeStyle = function(e){
  let target_style = document.getElementById('style-expand');
  if (target_style.disabled){
    target_style.disabled = false;
    e.textContent = "Shrink";
    e.classList.add("btn-shrink");
    e.classList.remove("btn-expand");
  }else{
    target_style.disabled = true;
    e.textContent = "Expand";
    e.classList.add("btn-expand");
    e.classList.remove("btn-shrink");
  }
  return false;
}
