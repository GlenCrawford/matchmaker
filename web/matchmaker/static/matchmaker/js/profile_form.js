$(document).ready(function() {
  var loading_interval;

  $('form#profile_form').submit(function(event) {
    var $form = $(event.target);

    $.ajax({
      method: $form.prop('method'),
      url: $form.prop('action'),
      dataType: 'html',
      data: $form.serialize(),
      beforeSend: function(jqXHR, settings) {
        $('html, body').animate({ scrollTop: 0 }, 500);

        $('#waiting-container').addClass('d-none');
        $('#loading-container').text('Finding your Mr. Darcy.').removeClass('d-none');
        $('#matches-container').addClass('d-none');
        $('#error-container').addClass('d-none');

        loading_interval = setInterval(function() {
          $('#loading-container').append('.')
        }, 1000);
      }
    }).done(function(data, textStatus, jqXHR) {
      $('#loading-container').addClass('d-none');
      $('#matches-container').removeClass('d-none').html(data);
      $('[data-toggle="popover"]').popover();

      clearInterval(loading_interval);
    })
    .fail(function(jqXHR, textStatus, errorThrown) {
      if (jqXHR.status == 422) {
        var humanized_validation_errors = [];
        $.each(JSON.parse(jqXHR.responseText), function(field, errors) {
          $.each(errors, function(index, error) {
            humanized_validation_errors.push(field + ': ' + error['message']);
          });
        });
        $('#error-container').html('Form errors:<br />' + humanized_validation_errors.join('<br />'));
      }
      else {
        $('#error-container').html('Error :(');
      }

      $('#loading-container').addClass('d-none');
      $('#error-container').removeClass('d-none');

      clearInterval(loading_interval);
    });

    event.preventDefault();
  });
});
