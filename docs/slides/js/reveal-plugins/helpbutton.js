var RevealHelpButton = ( function( _Reveal ){

	var Reveal = _Reveal;
	var defMode = 'first';

	function installHelpButtonDom( event ){
		// install help button in lower left position but initially
		// hide it until all styles are set
		var control = document.createElement( 'aside' );
		control.classList.add( 'controls', 'helpbutton' );
		control.innerHTML = '<button class="show-help" aria-label="show help" disabled="disabled" style="display: none;" onclick="Reveal.toggleHelp()"><div class="controls-help"></div></button>';
		document.querySelector( '.reveal' ).appendChild( control );
	}

	function setHelpButtonConfig( o ){
		var values = [ 'none', 'first', 'intro', 'always' ];
		var config = Reveal.getConfig();
		var helpButtonDisplay = o===Object(o) && o.helpButtonDisplay ? o.helpButtonDisplay
			: config.helpButtonDisplay ? config.helpButtonDisplay
			: defMode;
		if( values.indexOf( helpButtonDisplay ) < 0 ){
			console.error( 'Invalid value for configuration helpButtonDisplay. Allowed values are: ' + values.join( ', ') + '. Falling back to default ' + defMode + '.');
			helpButtonDisplay = defMode;
		}
		Reveal.configure({ helpButtonDisplay: helpButtonDisplay });
	}

	function setHelpButton( event ){
		// event.previousSlide, event.currentSlide, event.indexh, event.indexv
		var help = document.querySelector( '.show-help' );
		if( help ){
			var config = Reveal.getConfig();
			var display = 'none';
			if( config.helpButtonDisplay == 'always'
				|| ( config.helpButtonDisplay == 'intro' && !event.indexh )
				|| ( config.helpButtonDisplay == 'first' && !event.indexh && !event.indexv ) ){
				display = 'block';
			}
			if( help.style.display != display ){
				help.style.display = display;
			}
			if( display == 'none' ){
				help.classList.remove( 'enabled' );
				help.setAttribute( 'disabled', 'disabled' );
			}else{
				help.classList.add( 'enabled' );
				help.removeAttribute( 'disabled' );
			}
		}
	}

	function toggleHelpButton( event ){
		setHelpButton({
			previousSlide: Reveal.getPreviousSlide(),
			currentSlide: Reveal.getCurrentSlide(),
			indexh: Reveal.getIndices().h,
			indexv: Reveal.getIndices().v
		});
	}

	function installHelpButton(){
		installHelpButtonDom();

		// the slidechange event will not be triggered on
		// initial page load, so we need to toggle the
		// button manually here once
		Reveal.addEventListener( 'slidechanged', setHelpButton );
		toggleHelpButton();
	}

	function configure( o ){
		setHelpButtonConfig( o );
		toggleHelpButton();
	}

	function install(){
		// See https://iwasakishuto.github.io/js/custom.js
		var curt_path = getCurtPath();
		if (curt_path){
			curt_path = curt_path.replace(".js", ".css");
		}else{
			curt_path = Reveal.getConfig().pluginBaseDirectory + "helpbutton.css";
		}
		addStylesheet(curt_path);
		setHelpButtonConfig();
		// in case we are loading async the ready event may already
		// been emitted, so test here for readiness here and install
		// manually here
		if( Reveal.isReady() ){
			installHelpButton();
		}else{
			Reveal.addEventListener( 'ready', function(){
				installHelpButton();
			});
		}
	}

	var Plugin = {
		configure: configure
	}

	if( Reveal && Reveal.VERSION && Reveal.VERSION.length && Reveal.VERSION[ 0 ] == '3' ){
		// reveal 3.x
		install();
	}else{
		// must be reveal 4.x
		Plugin.id = 'help-button';
		Plugin.init = function( _Reveal ){
			Reveal = _Reveal;
			install();
		};
	}

	return Plugin;

})( Reveal );