function isFunction(func) {
  return func && {}.toString.call(func) === '[object Function]';
 }
// Update the "window.onload"
function addOnLoad(fn){
  if (isFunction(fn)){
    var old = window.onload;
    window.onload = isFunction(old) 
      ? function(){old();fn();} 
      : fn
  }
}

// 
(function (){
  $(function(){
    var items = document.querySelectorAll('div.toctree-wrapper.compound li[class*="toctree-"]');
    items.forEach((item) => {
      var text  = item.querySelector("a").innerHTML
      var text_components = text.split(".");
      var num_components = text_components.length;
      if (num_components>0){
        text = text_components[num_components-1];
      }
      text = text.replace(/(.*)\spackage/g,' <span class="package-name">$1 package</span>')
                .replace(/(.*)\smodule/g, '<span class="program-name">$1.py</span>')
                .replace(/(Subpackages|Submodules)/g,'<span class="package-subtitle">$1</span>')
                .replace(/(Module\scontents)/g, '<span class="module-contents">$1</span>');
      a = item.querySelector("a")
      a.innerHTML = text
      if (!text.includes("<span")) a.classList.add("chapter")
    });
  });
})(jQuery);

// Initialization for Stating Moving Background Particles
const initializeParticles = function(){
  Particles.init({
    selector: '.particle-background',
    color: ["#cee6b4", "#1f441e"],
    maxParticles: 0,
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
  Particles._resize();
}
addOnLoad(initializeParticles);
// <--- [Utility Functions (Button)] ---
// Start & Stop the Particles Movings.
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
// Change the display style (width)
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
// --- [Utility Functions (Button)]  --->

const instructions2details = function(){
  // [Before]
  // <div class="section" id="instructions">
  //   <h2>Instructions</h2>
  // </div>
  // 
  // [After]
  // <details class="section" id="instructions">
  //   <summary><h2>Instructions</h2></summary>
  // </details>
  $('div.section#instructions').replaceWith(function(){
    $(this).replaceWith('<details class="section" id="instructions" open>'+$(this).html()+"</details>")
  });
  $('details.section#instructions h2').wrap($('<summary></summary>'))
}
addOnLoad(instructions2details);