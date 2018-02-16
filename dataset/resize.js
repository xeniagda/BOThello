var ImageResize = require('node-image-resize'),
    fs = require('fs');
 
var im = require('imagemagick');

for (var i = 0; i < 61; i++) {
    (function(){
        var file = 'game1/img'+i + ".jpg";
        im.resize({
          srcPath: file,
          dstPath: file,
          height: 128
        }, function(err, stdout, stderr){
            im.convert([file, '-resize', '128x128','-gravity', 'center', '-background', 'black','-extent','128x128',file],function(err, stdout, stderr){
              // foo 
            });
        });
    })();
}