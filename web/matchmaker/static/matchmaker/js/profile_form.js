$(document).ready(function() {
  $('form#profile_form').submit(function(event) {
    var $form = $(event.target);

    $.ajax({
      method: $form.prop('method'),
      url: $form.prop('action'),
      dataType: 'html',
      data: $form.serialize(),
      beforeSend: function(jqXHR, settings) {
        window.scrollTo({ top: 0, behavior: 'smooth' });

        $('#waiting-container').addClass('d-none');
        $('#loading-container').removeClass('d-none');
        $('#matches-container').addClass('d-none');
        $('#error-container').addClass('d-none');
      }
    }).done(function(data, textStatus, jqXHR) {
      $('#loading-container').addClass('d-none');
      $('#matches-container').removeClass('d-none').html(data);
    })
    .fail(function(jqXHR, textStatus, errorThrown) {
      $('#loading-container').addClass('d-none');
      $('#error-container').removeClass('d-none');
    });

    event.preventDefault();
  });
});
