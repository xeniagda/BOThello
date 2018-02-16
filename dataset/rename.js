var fs = require("fs");
var exec = require('child_process').exec;
var files = fs.readdirSync("./");

var parsedFiles = [];


var pre = "game1/imgx";
var suf = ".jpg";
var newPre = "game1/img";
var newSuf = ".jpg";

for (var i = 0; i < files.length; i++) {
	var reg = RegExp(pre+"([0-9.]+)"+suf);
	if(files[i].match(reg)){
		parsedFiles.push(parseFloat(files[i].match(reg)[1]));
	}
}

parsedFiles = parsedFiles.sort((a,b)=>a-b);
console.log(parsedFiles);

for (var i = 0; i < parsedFiles.length; i++) {
	exec("mv " + pre + parsedFiles[i] + suf + " " + newPre + i + newSuf); 
}