
function show_details(){
     d = document.getElementById("queryid").value;
     document.getElementById("description").innerHTML  =  "/* " +document.getElementById(d+"_d").value + "*/"
     document.getElementById("body").src  = '/static/images/'+d+'.png'//document.getElementById(d+"_b").value
     document.getElementById("query").value = d

}


function click_val() {
  var truth_val = $("#value").val()
  var timer = $("#idx").val() < 10 ? 1000 : 3000
  if (truth_val == "1"){
//

//      $( "#t_ans" ).click(function() {
        $( "#t_ans" ).animate({
            opacity: 0.35,
            left: "+=50"
          }, 2000, function() {
            // Animation complete.
          });
//        });
    setTimeout(function(){ $('#t_ans').trigger('click'); }, timer);

  }
  else{
//
//    $( "#f_ans" ).click(function() {
        $( "#f_ans" ).animate({
            opacity: 0.35,
            left: "+=50"
          }, 2000, function() {
            // Animation complete.
          });
//        });
    setTimeout(function(){ $('#f_ans').trigger('click'); }, timer);
    //$('#f_ans').trigger('click');
  }
}

$(document).ready(function(){
    $( "#pause" ).click(function() {
        if ($( "#auto").val() == "True"){
            $( "#auto").val("False")
            $( "#pause" ).html("Auto. Evaluation")
        }
        else{
            $( "#auto").val("True")
            $( "#pause").html("Pause Evaluation")
            click_val()
        }
    })
  var auto = $("#auto").val()

  if (typeof auto !== 'undefined' && auto == 'True'){
    setTimeout(click_val, 1000);
  }

});


